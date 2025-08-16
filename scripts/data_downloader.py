#!/usr/bin/env python3
import os, time, logging, argparse, random, csv, gzip, threading
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Defaults tuned for Jan→Jun 2025 ---
DEFAULT_PRIMARY_URL = "https://api.binance.com/api/v3/aggTrades"
DEFAULT_FALLBACK_URL = "https://data-api.binance.vision/api/v3/aggTrades"
DEFAULT_DATA_DIR = "./tick_data"
DEFAULT_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_START = "2025-01-01"
DEFAULT_END   = "2025-06-30"

PAGE_LIMIT = 1000
MAX_RETRIES = 6
BASE_RETRY_DELAY = 1.0
MAX_PAGES_PER_HOUR = 10000

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("binance-ticks")

# --- Helpers ---
def to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def jitter(seconds: float) -> float:
    return seconds * (0.8 + 0.4 * random.random())

def _safe_unique_tmp_path(final_path: Path) -> Path:
    pid = os.getpid()
    tid = threading.get_ident()
    ts = int(time.time() * 1000)
    return final_path.with_name(final_path.name + f".tmp.{pid}.{tid}.{ts}")

def _ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()

# --- HTTP client with failover + backoff ---
class HttpClient:
    def __init__(self, primary: str, fallback: str, timeout: int = 30):
        self.primary = primary
        self.fallback = fallback
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "tsmom-downloader/1.0"})
        adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=0)
        self.s.mount("https://", adapter)
        self.s.mount("http://", adapter)

    def get(self, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        urls = [self.primary, self.fallback] if self.fallback else [self.primary]
        last_err = None
        for url in urls:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    r = self.s.get(url, params=params, timeout=self.timeout)
                    if r.status_code == 429:
                        wait_hdr = r.headers.get("Retry-After")
                        wait = float(wait_hdr) if wait_hdr else jitter(BASE_RETRY_DELAY * (2 ** (attempt - 1)))
                        log.warning("429 rate limited @ %s. Sleeping %.2fs (attempt %d/%d)", url, wait, attempt, MAX_RETRIES)
                        time.sleep(wait)
                        continue
                    r.raise_for_status()
                    data = r.json()
                    if not isinstance(data, list):
                        raise requests.RequestException(f"Unexpected payload type: {type(data)} {data}")
                    return data
                except Exception as e:
                    last_err = e
                    wait = jitter(BASE_RETRY_DELAY * (2 ** (attempt - 1)))
                    log.warning("GET error on %s (attempt %d/%d): %s — sleeping %.2fs",
                                url, attempt, MAX_RETRIES, e, wait)
                    time.sleep(wait)
            log.warning("Switching to fallback/next URL after repeated errors: %s", url)
        log.error("All URLs failed. Last error: %s", last_err)
        return None

# --- Downloader ---
class BinanceAggTradeFetcher:
    def __init__(self, symbol: str, data_dir: str, client: HttpClient, compress: str = "none", force: bool = False, verbose: bool = False):
        self.symbol = symbol.upper()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.compress = compress  # "none" | "gzip"
        self.force = force
        self.verbose = verbose

    def _month_outpath(self, month_start: datetime) -> Path:
        fname = f"{self.symbol}-ticks-{month_start.strftime('%Y-%m')}.csv"
        if self.compress == "gzip":
            fname += ".gz"
        return self.data_dir / fname

    def _first_page_by_time(self, start_ms: int, end_ms_excl: int) -> List[Dict[str, Any]]:
        params = {"symbol": self.symbol, "startTime": start_ms, "endTime": end_ms_excl, "limit": PAGE_LIMIT}
        return self.client.get(params) or []

    def _next_page_by_id(self, from_id: int) -> List[Dict[str, Any]]:
        params = {"symbol": self.symbol, "fromId": from_id, "limit": PAGE_LIMIT}
        return self.client.get(params) or []

    def _writer(self, out_fh, gz: bool):
        import io
        gzfh = None
        if gz:
            gzfh = gzip.GzipFile(fileobj=out_fh, mode="wb")
            txt = io.TextIOWrapper(gzfh, encoding="utf-8", newline="")
        else:
            txt = io.TextIOWrapper(out_fh, encoding="utf-8", newline="")
        w = csv.writer(txt)
        w.writerow(["timestamp", "price", "qty", "is_buyer_maker"])
        def write_row(ts, price, qty, m):
            w.writerow([ts, f"{price}", f"{qty}", 1 if m else 0])
        def close():
            txt.flush()
            if gz:
                txt.detach()
                gzfh.close()
            out_fh.close()
        return write_row, close

    def _process_hour(self, start_ms: int, end_ms_excl: int, write_row) -> int:
        written = 0
        page = self._first_page_by_time(start_ms, end_ms_excl)
        seen_ids = set()
        pages = 0
        while page:
            pages += 1
            for r in page:
                aid = r.get("a")
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)
                ts = int(r["T"])
                if ts >= end_ms_excl:
                    if self.verbose:
                        log.info("[%s] Hour %s→%s reached end_ms_excl early; rows=%d pages=%d",
                                 self.symbol, _ms_to_iso(start_ms), _ms_to_iso(end_ms_excl), written, pages)
                    return written
                price = float(r["p"]); qty = float(r["q"]); m = bool(r["m"])
                write_row(ts, price, qty, m)
                written += 1
            if len(page) < PAGE_LIMIT:
                break
            if pages >= MAX_PAGES_PER_HOUR:
                log.warning("[%s] Safety stop: exceeded MAX_PAGES_PER_HOUR in hour starting %s",
                            self.symbol, _ms_to_iso(start_ms))
                break
            last_id = int(page[-1]["a"])
            page = self._next_page_by_id(last_id + 1)
        if self.verbose:
            log.info("[%s] Hour %s→%s done; rows=%d pages=%d",
                     self.symbol, _ms_to_iso(start_ms), _ms_to_iso(end_ms_excl), written, pages)
        return written

    def _process_month(self, month_start: datetime, month_end_incl: datetime):
        final_path = self._month_outpath(month_start)
        if final_path.exists() and not self.force:
            log.info("[%s] %s exists. Skipping.", self.symbol, final_path.name)
            return

        tmp_path = _safe_unique_tmp_path(final_path)
        month_label = month_start.strftime("%Y-%m")
        log.info("[%s] Downloading month %s …", self.symbol, month_label)

        start_ms = to_ms(month_start)
        end_excl = to_ms((month_end_incl + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc))

        total_rows = 0
        hour_ms = 60 * 60 * 1000
        total_hours = max(1, (end_excl - start_ms) // hour_ms)

        with open(tmp_path, "wb") as raw_fh:
            write_row, close_writer = self._writer(raw_fh, gz=(self.compress == "gzip"))
            # Hour-level progress bar so you SEE work happening
            with tqdm(total=total_hours,
                      desc=f"{self.symbol} {month_label}",
                      unit="h",
                      leave=True,
                      dynamic_ncols=True) as hourbar:
                cur = start_ms
                while cur < end_excl:
                    next_hour = min(cur + hour_ms, end_excl)
                    total_rows += self._process_hour(cur, next_hour, write_row)
                    cur = next_hour
                    hourbar.update(1)
            close_writer()

        if total_rows == 0:
            log.warning("[%s] No data for %s. Removing empty file.", self.symbol, month_label)
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            return

        os.replace(tmp_path, final_path)  # atomic finalize
        log.info("[%s] Saved %d rows -> %s", self.symbol, total_rows, final_path.name)

    def run(self, start_date: datetime, end_date: datetime):
        start_date = start_date.replace(tzinfo=timezone.utc, hour=0, minute=0, second=0, microsecond=0)
        end_date = end_date.replace(tzinfo=timezone.utc, hour=0, minute=0, second=0, microsecond=0)
        if end_date < start_date:
            raise ValueError("end_date must be >= start_date")

        total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
        cur = start_date.replace(day=1)
        # Month-level bar (increments after each month completes)
        with tqdm(total=total_months,
                  desc=f"{self.symbol} months",
                  unit="mo",
                  leave=True,
                  dynamic_ncols=True) as monthbar:
            while cur <= end_date:
                nxt = (cur + timedelta(days=32)).replace(day=1)
                month_end = min(nxt - timedelta(days=1), end_date)
                self._process_month(cur, month_end)
                cur = nxt
                monthbar.update(1)

# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser(description="Download Binance aggTrades to monthly CSV(.gz).")
    p.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="Trading pairs to download")
    p.add_argument("--start-date", default=DEFAULT_START, help="Start date YYYY-MM-DD UTC")
    p.add_argument("--end-date",   default=DEFAULT_END,   help="End date YYYY-MM-DD UTC (inclusive)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Output directory")
    p.add_argument("--primary-url", default=DEFAULT_PRIMARY_URL, help="Primary API URL")
    p.add_argument("--fallback-url", default=DEFAULT_FALLBACK_URL, help="Fallback API URL")
    p.add_argument("--compress", choices=["none", "gzip"], default="none", help="Output compression format")
    p.add_argument("--max-workers", type=int, default=3, help="Concurrent symbol downloads")
    p.add_argument("--force", action="store_true", help="Re-download even if file exists")
    p.add_argument("--verbose", action="store_true", help="Log per-hour stats while downloading")
    p.add_argument("--jan-jun-2025", action="store_true",
                   help="Shortcut: sets --start-date 2025-01-01 --end-date 2025-06-30")
    return p.parse_args()

def main():
    args = parse_args()

    # argparse converts dashes to underscores → --jan-jun-2025 -> args.jan_jun_2025
    if getattr(args, "jan_jun_2025", False):
        args.start_date, args.end_date = DEFAULT_START, DEFAULT_END

    try:
        sdt = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        edt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise SystemExit(f"Bad date: {e}")

    client = HttpClient(primary=args.primary_url, fallback=args.fallback_url, timeout=30)
    os.makedirs(args.data_dir, exist_ok=True)

    log.info("Config: pairs=%s start=%s end=%s out=%s compress=%s verbose=%s",
             args.pairs, args.start_date, args.end_date, args.data_dir, args.compress, args.verbose)

    def worker(sym: str):
        log.info("[%s] Starting download window %s → %s", sym, args.start_date, args.end_date)
        fetcher = BinanceAggTradeFetcher(sym, args.data_dir, client,
                                         compress=args.compress, force=args.force, verbose=args.verbose)
        fetcher.run(sdt, edt)
        log.info("[%s] Finished.", sym)

    # Concurrency across symbols
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(worker, sym): sym for sym in args.pairs}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                fut.result()
                log.info("[%s] ALL MONTHS DOWNLOADED", sym)
            except Exception as e:
                log.error("[%s] FAILED: %s", sym, e)

    log.info("All downloads complete.")

if __name__ == "__main__":
    main()
