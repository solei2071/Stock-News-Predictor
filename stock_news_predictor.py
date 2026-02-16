#!/usr/bin/env python3
"""
stock_news_predictor.py

Python GUI app that fetches company news from public RSS sources (no Yahoo news query)
plus recent prices, then visualizes a lightweight news/price based forecast for a stock ticker.
"""

from __future__ import annotations

import math
import re
import statistics
import ssl
from datetime import datetime, timezone, timedelta
from html import unescape
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import threading
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

USER_AGENT = "Mozilla/5.0 (compatible; StockNewsPredictor/1.1)"
NEWS_CRAWLING_ENABLED = False

POSITIVE = {
    "beat": 2,
    "surge": 2,
    "surged": 2,
    "rally": 2,
    "rallies": 2,
    "growth": 2,
    "record": 1,
    "upgrade": 2,
    "upside": 1,
    "profit": 1,
    "strong": 1,
    "expansion": 1,
    "improve": 1,
    "improved": 1,
    "success": 1,
    "buy": 1,
    "bullish": 2,
    "innovation": 1,
    "혁신": 2,
    "성장": 2,
    "호재": 2,
    "상승": 2,
    "개선": 1,
    "흑자": 1,
}

NEGATIVE = {
    "miss": 2,
    "missed": 2,
    "drop": 2,
    "dropped": 2,
    "fall": 2,
    "fell": 2,
    "loss": 2,
    "losses": 2,
    "downgrade": 2,
    "warning": 1,
    "weak": 1,
    "weakness": 1,
    "risk": 1,
    "lawsuit": 2,
    "investigation": 1,
    "recession": 2,
    "decline": 2,
    "downside": 1,
    "bearish": 2,
    "부진": 2,
    "감소": 2,
    "적자": 2,
    "우려": 1,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _open_url(url: str, timeout: int = 20) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=timeout) as response:
            return response.read()
    except (ssl.SSLError, URLError):
        # macOS Python/CA 번들 문제 대응: 필요 시 보안 검증을 비활성화하고 한 번 재시도
        insecure_ctx = ssl._create_unverified_context()
        with urlopen(req, timeout=timeout, context=insecure_ctx) as response:
            return response.read()


def fetch_json(url: str, timeout: int = 20) -> Dict:
    text = _open_url(url, timeout=timeout).decode("utf-8", errors="ignore")
    return __import__("json").loads(text)


def parse_price_data(symbol: str, period: str = "6mo") -> Dict:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(symbol)}?interval=1d&range={quote(period)}&includeAdjustedClose=true"
    )
    payload = fetch_json(url)

    results = payload.get("chart", {}).get("result", []) if isinstance(payload, dict) else []
    if not results:
        raise RuntimeError("주가 데이터가 없습니다.")

    result = results[0]
    timestamps = result.get("timestamp", []) or []
    indicators = result.get("indicators", {}).get("quote", [])
    if not indicators:
        raise RuntimeError("가격 지표가 없습니다.")

    closes = indicators[0].get("close", []) or []
    opens = indicators[0].get("open", []) or []
    highs = indicators[0].get("high", []) or []
    lows = indicators[0].get("low", []) or []
    if not timestamps or not closes:
        raise RuntimeError("종가 데이터가 비어 있습니다.")

    rows: List[Tuple[datetime, float, float, float, float]] = []
    for i, ts in enumerate(timestamps):
        if i >= len(closes):
            break

        close = closes[i]
        if close is None:
            continue

        open_price = opens[i] if i < len(opens) and opens[i] is not None else close
        high_price = highs[i] if i < len(highs) and highs[i] is not None else close
        low_price = lows[i] if i < len(lows) and lows[i] is not None else close
        if open_price is None or high_price is None or low_price is None:
            continue
        try:
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except Exception:
            continue
        rows.append((dt, float(close), float(open_price), float(high_price), float(low_price)))

    if not rows:
        raise RuntimeError("유효한 가격 데이터가 없습니다.")

    meta = result.get("meta", {}) or {}
    return {
        "rows": rows,
        "symbol": meta.get("symbol", symbol),
        "name": meta.get("longName") or meta.get("shortName") or symbol,
        "currency": meta.get("currency", "USD"),
        "exchange": meta.get("exchangeName", "N/A"),
        "current_price": float(meta.get("regularMarketPrice") or rows[-1][1]),
    }


def _coerce_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1_000_000_000_000:
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.isdigit():
            return _coerce_datetime(float(value))
    for fm in ("%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            parsed = datetime.strptime(value, fm)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return None


def _looks_like_ticker(query: str) -> bool:
    q = query.strip()
    if not q or " " in q:
        return False
    if "." in q:
        prefix, suffix = q.split(".", 1)
        if not prefix or not suffix:
            return False
    if "-" in q:
        prefix, suffix = q.split("-", 1)
        if not prefix or not suffix:
            return False
    if len(q) > 12:
        return False
    core = q.replace(".", "").replace("-", "")
    if not re.fullmatch(r"[A-Za-z0-9]+", core):
        return False
    return q.isupper() and 1 <= len(q) <= 6


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z가-힣]+", " ", (text or "").lower())).strip()


def _match_score(query: str, text: str) -> int:
    if not query or not text:
        return 0
    if query in text:
        return 3
    q_parts = query.split()
    if q_parts and all(part in text for part in q_parts):
        return 2
    return 1 if any(part in text for part in q_parts) else 0


def resolve_symbol_by_name(query: str) -> str:
    q = query.strip()
    if not q:
        raise RuntimeError("종목명/티커를 입력해주세요.")

    url = (
        "https://query2.finance.yahoo.com/v1/finance/search?"
        f"q={quote(q)}&quotesCount=10&newsCount=0&listsCount=0"
    )
    if _looks_like_ticker(q):
        candidate = q.upper()
        try:
            payload = fetch_json(url)
            if isinstance(payload, dict):
                for item in payload.get("quotes", []) or []:
                    if isinstance(item, dict) and (item.get("symbol") or "").upper() == candidate:
                        return candidate
        except Exception:
            # fallback to full search logic below
            pass

    payload = fetch_json(url)
    if not isinstance(payload, dict):
        raise RuntimeError("검색 응답 형식이 올바르지 않습니다.")

    quotes = payload.get("quotes", [])
    if not quotes:
        raise RuntimeError(f"입력값 '{q}'에 해당하는 종목을 찾지 못했습니다.")

    target = _normalize_text(q)
    fallback_symbol = None
    best_symbol = None
    best_score = 0
    for item in quotes:
        if not isinstance(item, dict):
            continue
        if item.get("quoteType", "EQUITY").upper() not in {"EQUITY", "ETF"}:
            continue

        short_name = _normalize_text(item.get("shortName") or "")
        long_name = _normalize_text(item.get("longName") or "")
        name = _normalize_text(item.get("name") or "")
        symbol = (item.get("symbol") or "").strip()
        if not symbol:
            continue

        if fallback_symbol is None:
            fallback_symbol = symbol

        score = max(
            _match_score(target, short_name),
            _match_score(target, long_name),
            _match_score(target, name),
        )
        if score > best_score:
            best_score = score
            best_symbol = symbol
            if score >= 3:
                return symbol.upper()

    if best_symbol:
        return best_symbol.upper()

    if fallback_symbol:
        return fallback_symbol.upper()

    raise RuntimeError(f"입력값 '{q}'에 해당하는 종목을 찾지 못했습니다.")


def _add_trading_days(base_date: datetime, count: int) -> datetime:
    if count <= 0:
        return base_date
    dt = base_date
    remain = count
    while remain > 0:
        dt = dt + timedelta(days=1)
        if dt.weekday() < 5:
            remain -= 1
    return dt


def _strip_html(text: str) -> str:
    clean = re.sub(r"<[^>]*?>", " ", unescape(text or ""))
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _first_text(node: ET.Element, names: List[str]) -> str:
    for name in names:
        child = node.find(name)
        if child is not None and isinstance(child.text, str) and child.text.strip():
            return child.text.strip()
        if child is not None and isinstance(child.text, str):
            return child.text.strip()
    return ""


def _extract_rss_items(url: str, provider: str, limit: int) -> List[Dict]:
    xml_raw = _open_url(url, timeout=25).decode("utf-8", errors="ignore")

    root = ET.fromstring(xml_raw)
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    parsed: List[Dict] = []
    for node in items:
        title = _strip_html(_first_text(node, ["title", "{http://www.w3.org/2005/Atom}title"]))
        if not title:
            continue

        link = _first_text(node, ["link", "{http://www.w3.org/2005/Atom}link"])
        if not link:
            for link_node in node.findall("link") + node.findall("{http://www.w3.org/2005/Atom}link"):
                if not isinstance(link_node.tag, str):
                    continue
                if link_node.text:
                    link = link_node.text.strip()
                    break
                href = link_node.attrib.get("href") if hasattr(link_node, "attrib") else None
                if href:
                    link = href.strip()
                    break

        summary = _strip_html(
            _first_text(node, ["description", "{http://www.w3.org/2005/Atom}summary"]) or ""
        )
        pub = _coerce_datetime(
            _first_text(node, ["pubDate", "published", "updated", "{http://www.w3.org/2005/Atom}published", "{http://www.w3.org/2005/Atom}updated"])
        )

        source = _first_text(node, ["source", "{http://www.w3.org/2005/Atom}source"])
        if not source:
            source = provider

        parsed.append(
            {
                "title": title,
                "link": link,
                "summary": summary,
                "provider": source,
                "published": pub,
            }
        )

        if len(parsed) >= limit:
            break

    return parsed


def fetch_news(symbol: str, limit: int = 20) -> List[Dict]:
    # 뉴스 크롤링은 현재 비활성화됨: 차단 이슈 방지를 위해 임시 중단
    if not NEWS_CRAWLING_ENABLED:
        return []

    queries = [
        f"{symbol} stock",
        f"{symbol} company",
        f"{symbol}",
    ]
    providers = [
        (
            "Google News",
            lambda q: f"https://news.google.com/rss/search?q={quote(q)}&hl=ko&gl=KR&ceid=KR:ko",
        ),
        (
            "Bing News",
            lambda q: f"https://www.bing.com/news/search?q={quote(q)}&format=rss",
        ),
    ]

    collected: List[Dict] = []
    dedup = set()
    last_error: Optional[Exception] = None

    for q in queries:
        for provider, builder in providers:
            try:
                crawled = _extract_rss_items(builder(q), provider, limit)
            except Exception as exc:  # pragma: no cover
                last_error = exc
                continue

            for item in crawled:
                key = (item.get("link") or item.get("title") or "").strip()
                if not key or key in dedup:
                    continue
                dedup.add(key)
                item["provider"] = provider if not item.get("provider") else item["provider"]
                collected.append(item)

            if len(collected) >= limit:
                break
        if len(collected) >= limit:
            break

    if not collected:
        err = f"{last_error}" if last_error else "no item parsed"
        raise RuntimeError(f"뉴스 수집에 실패했습니다: {err}")

    collected.sort(key=lambda n: n["published"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return collected[:limit]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z가-힣]+", (text or "").lower())


def sentiment_score(items: List[Dict]) -> Tuple[float, List[Tuple[str, float]]]:
    scores: List[Tuple[str, float]] = []
    if not items:
        return 0.0, scores

    total = 0.0
    for item in items:
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        tokens = tokenize(text)
        if not tokens:
            per_score = 0.0
        else:
            per_score = 0.0
            for token in tokens:
                per_score += POSITIVE.get(token, 0)
                per_score -= NEGATIVE.get(token, 0)
            per_score = clamp(per_score / len(tokens), -1.0, 1.0)

        total += per_score
        scores.append((item.get("title", ""), per_score))

    return clamp(total / len(items), -1.0, 1.0), scores


def linear_forecast(prices: List[float], horizon: int) -> float:
    n = len(prices)
    if n < 3 or horizon <= 0:
        return 0.0

    xs = list(range(n))
    ys = [math.log(p) for p in prices]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)

    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x

    forecast_x = xs[-1] + horizon
    forecast_log = intercept + slope * forecast_x
    return (math.exp(forecast_log - math.log(prices[-1])) - 1.0) * 100.0


def volatility(prices: List[float]) -> float:
    if len(prices) < 2:
        return 0.25

    rets = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        curr = prices[i]
        if prev <= 0:
            continue
        rets.append(math.log(curr / prev))

    if len(rets) < 2:
        return 0.25

    return statistics.pstdev(rets)


def format_currency(value: float, currency: str) -> str:
    if currency == "USD":
        return f"${value:,.2f}"
    return f"{value:,.2f} {currency}"


def build_analysis(symbol: str, horizon: int, news_limit: int, period: str) -> Dict:
    price_info = parse_price_data(symbol, period)
    rows = price_info["rows"]
    if not rows:
        raise RuntimeError("가격 데이터가 없습니다.")

    prices = [close for _, close, _, _, _ in rows]
    dates = [dt for dt, _, _, _, _ in rows]
    candle_data = [(open_price, high_price, low_price, close_price) for _, close_price, open_price, high_price, low_price in rows]
    current_price = prices[-1]

    news = fetch_news(symbol, limit=news_limit)
    sentiment, per_item_scores = sentiment_score(news)

    if NEWS_CRAWLING_ENABLED and news:
        sentiment_label = "중립"
        if sentiment > 0.12:
            sentiment_label = "긍정"
        elif sentiment < -0.12:
            sentiment_label = "부정"
    else:
        sentiment_label = "중립(미적용)"

    trend_pct = linear_forecast(prices, horizon)
    sentiment_adj = sentiment * min(max(horizon, 1) / 7.0, 2.5) * 2.2
    final_pct = clamp(trend_pct + sentiment_adj, -40.0, 40.0)

    vol = volatility(prices)
    band = clamp(vol * 100 * math.sqrt(horizon), 1.5, 18.0)

    predicted = current_price * (1 + final_pct / 100.0)
    lower = current_price * (1 + (final_pct - band) / 100.0)
    upper = current_price * (1 + (final_pct + band) / 100.0)

    data_conf = clamp(len(prices) / 120.0 * 100.0, 0, 100)
    news_conf = clamp(len(news) / 20.0 * 100.0, 0, 100)
    trend_conf = clamp(72 - (abs(vol) * 140), 20, 75)
    confidence = clamp(15 + 0.45 * data_conf + 0.35 * news_conf + 0.20 * trend_conf, 10, 95)

    signal = "보합"
    if final_pct >= 1.2:
        signal = "매수 관망"
    elif final_pct <= -1.2:
        signal = "주의"

    score_by_title = {title: score for title, score in per_item_scores}
    news_preview: List[Dict] = []
    for item in news[:8]:
        title = item.get("title", "")
        news_preview.append(
            {
                "title": title,
                "provider": item.get("provider", "unknown"),
                "published": item.get("published"),
                "summary": str(item.get("summary", ""))[:140],
                "link": item.get("link", ""),
                "score": score_by_title.get(title, 0.0),
            }
        )

    return {
        "symbol": price_info["symbol"],
        "name": price_info["name"],
        "exchange": price_info["exchange"],
        "currency": price_info["currency"],
        "current_price": current_price,
        "latest_date": rows[-1][0].date().isoformat(),
        "dates": dates[-max(30, horizon + 10) :],
        "prices": prices[-max(30, horizon + 10) :],
        "candles": candle_data[-max(30, horizon + 10) :],
        "horizon": horizon,
        "trend_pct": trend_pct,
        "news_sentiment": sentiment,
        "sentiment_label": sentiment_label,
        "final_pct": final_pct,
        "predicted": predicted,
        "lower": lower,
        "upper": upper,
        "signal": signal,
        "confidence": confidence,
        "news_count": len(news),
        "news_items": news_preview,
        "news_enabled": NEWS_CRAWLING_ENABLED,
    }


class StockNewsPredictorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Stock News Predictor")
        self.geometry("1180x820")
        self.minsize(1020, 760)
        self.configure(bg="#0b1220")

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#0b1220")
        self.style.configure("TLabel", background="#0b1220", foreground="#e2e8f0")
        self.style.configure("Header.TLabel", font=("Malgun Gothic", 15, "bold"), foreground="#ffffff")
        self.style.configure("Title.TLabel", font=("Malgun Gothic", 24, "bold"), foreground="#60a5fa")
        self.style.configure("Value.TLabel", font=("Malgun Gothic", 18, "bold"), foreground="#f8fafc")
        self.style.configure("Subtle.TLabel", font=("Malgun Gothic", 10), foreground="#94a3b8")
        self.style.configure("TLabelframe", background="#101a33", foreground="#cbd5e1")
        self.style.configure("TLabelframe.Label", background="#101a33", foreground="#93c5fd")
        self.style.configure("TButton", background="#1e293b", foreground="#e2e8f0")
        self.style.configure("TEntry", fieldbackground="#0f172a", foreground="#e2e8f0")
        self.style.configure("TCombobox", fieldbackground="#0f172a", background="#0f172a", foreground="#e2e8f0")
        self.style.configure(
            "History.TCombobox",
            fieldbackground="#0f172a",
            background="#0f172a",
            foreground="#e2e8f0",
            selectbackground="#1e293b",
            selectforeground="#e2e8f0",
        )
        self.style.map(
            "History.TCombobox",
            fieldbackground=[("readonly", "#0f172a")],
            background=[("readonly", "#0f172a")],
            foreground=[("readonly", "#e2e8f0")],
            selectbackground=[("readonly", "#1e293b")],
            selectforeground=[("readonly", "#e2e8f0")],
        )
        self.style.configure(
            "News.Treeview",
            rowheight=22,
            font=("Malgun Gothic", 9),
            background="#111a33",
            fieldbackground="#111a33",
            foreground="#e2e8f0",
            borderwidth=0,
            lightcolor="#111a33",
            darkcolor="#111a33",
        )
        self.style.configure(
            "News.Treeview.Heading",
            font=("Malgun Gothic", 9, "bold"),
            background="#1e293b",
            foreground="#cbd5e1",
            relief="flat",
        )
        self.style.map(
            "News.Treeview",
            background=[("selected", "#1d4ed8")],
            foreground=[("selected", "#ffffff")],
            fieldbackground=[("selected", "#1d4ed8")],
        )
        self.style.map("News.Treeview.Heading", background=[("active", "#334155")], foreground=[("active", "#ffffff")])

        self.news_records: List[Dict] = []
        self._chart_points: List[Tuple[int, float, datetime, float, float, float, float]] = []
        # (x_left, x_right, y_top, y_bottom, min_price, max_price)
        self._chart_bounds: Tuple[int, int, int, int, float, float] = (0, 0, 0, 0, 0.0, 0.0)
        self._last_result: Optional[Dict] | None = None
        self._chart_resize_job: Optional[str] = None
        self._build_layout()

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="TFrame")
        root.pack(fill="both", expand=True, padx=14, pady=14)

        header = ttk.Frame(root, style="TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="종목 뉴스 기반 가격 예측", style="Title.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="현재 상태: 대기 중")
        ttk.Label(header, textvariable=self.status_var, style="Subtle.TLabel").pack(side="right")

        ctrl = ttk.LabelFrame(root, text="분석 설정", style="TLabelframe", padding=10)
        ctrl.pack(fill="x", pady=(12, 10))

        row1 = ttk.Frame(ctrl, style="TFrame")
        row1.pack(fill="x")

        ttk.Label(row1, text="종목/회사명", style="Subtle.TLabel").grid(row=0, column=0, padx=(0, 8), pady=4)
        self.symbol_var = tk.StringVar(value="AAPL")
        ttk.Entry(row1, textvariable=self.symbol_var, width=12).grid(row=0, column=1, padx=(0, 16), pady=4)

        ttk.Label(row1, text="기업명", style="Subtle.TLabel").grid(row=0, column=2, padx=(0, 8), pady=4)
        self.company_name_var = tk.StringVar(value="-")
        tk.Label(
            row1,
            textvariable=self.company_name_var,
            width=24,
            anchor="w",
            background="#0b1220",
            foreground="#f8fafc",
            font=("Malgun Gothic", 10),
        ).grid(row=0, column=3, padx=(0, 16), pady=4)

        ttk.Label(row1, text="예측 영업일", style="Subtle.TLabel").grid(row=0, column=4, padx=(0, 8), pady=4)
        self.days_var = tk.StringVar(value="7")
        ttk.Entry(row1, textvariable=self.days_var, width=6).grid(row=0, column=5, padx=(0, 16), pady=4)

        ttk.Label(row1, text="가격 이력", style="Subtle.TLabel").grid(row=0, column=6, padx=(0, 8), pady=4)
        self.period_var = tk.StringVar(value="6mo")
        ttk.Combobox(
            row1,
            textvariable=self.period_var,
            values=["1mo", "3mo", "6mo", "1y", "2y"],
            state="readonly",
            width=8,
            style="History.TCombobox",
        ).grid(row=0, column=7, padx=(0, 16), pady=4)

        self.run_btn = ttk.Button(row1, text="예측 실행", command=self._on_run)
        self.run_btn.grid(row=0, column=8, padx=(20, 0), pady=4)

        for i in range(9):
            row1.columnconfigure(i, weight=0)

        cards = ttk.Frame(root, style="TFrame")
        cards.pack(fill="x", pady=(4, 10))

        self._card_widgets = {}
        labels = [
            ("current", "현재가", "-", "-"),
            ("trend", "가격 추세", "-", "-"),
            ("sentiment", "뉴스 감성", "-", "-"),
            ("forecast", "예측 수익률", "-", "-"),
            ("confidence", "신뢰도", "-", "-"),
            ("range", "예측 구간", "-", "-"),
        ]

        for idx, (k, title, value, sub) in enumerate(labels):
            f = tk.Frame(cards, bg="#111a33", padx=14, pady=10)
            f.grid(row=0, column=idx, padx=6, sticky="nsew")
            cards.columnconfigure(idx, weight=1)
            tk.Label(f, text=title, fg="#94a3b8", bg="#111a33", font=("Malgun Gothic", 10, "bold")).pack(anchor="w")
            val = tk.Label(f, text=value, fg="#e2e8f0", bg="#111a33", font=("Malgun Gothic", 22, "bold"))
            val.pack(anchor="w", pady=(6, 0))
            sub_label = tk.Label(f, text=sub, fg="#cbd5e1", bg="#111a33", font=("Malgun Gothic", 9))
            sub_label.pack(anchor="w")
            self._card_widgets[k] = (val, sub_label)

        content = ttk.Frame(root, style="TFrame")
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=4)
        content.rowconfigure(1, weight=1)

        chart_frame = ttk.LabelFrame(content, text="가격 트렌드", style="TLabelframe", padding=8)
        chart_frame.grid(row=0, column=0, sticky="nsew")
        self.chart_canvas = tk.Canvas(chart_frame, bg="#0b1220", bd=0, highlightthickness=0)
        self.chart_canvas.pack(fill="both", expand=True)
        self.chart_tooltip = tk.Label(
            chart_frame,
            text="",
            bg="#0f172a",
            fg="#e2e8f0",
            bd=1,
            relief="solid",
            padx=8,
            pady=4,
            font=("Malgun Gothic", 9),
            justify="left",
        )
        self.chart_tooltip.place_forget()
        self.chart_canvas.bind("<Motion>", self._on_chart_hover)
        self.chart_canvas.bind("<Leave>", self._on_chart_leave)
        self.chart_canvas.bind("<Configure>", self._on_chart_resize)

        news_frame = ttk.LabelFrame(content, text="최신 뉴스", style="TLabelframe", padding=8)
        news_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        news_cols = ("시간", "출처", "감성", "제목")
        self.news_tree = ttk.Treeview(
            news_frame,
            columns=news_cols,
            show="headings",
            height=7,
            style="News.Treeview",
        )
        self.news_tree.pack(side="left", fill="both", expand=True)
        widths = {"시간": 130, "출처": 90, "감성": 50, "제목": 900}
        for c in news_cols:
            self.news_tree.heading(c, text=c)
            self.news_tree.column(c, width=widths[c], anchor="w")

        self.news_tree.bind("<Double-1>", self._open_news)
        news_scroll = ttk.Scrollbar(news_frame, orient="vertical", command=self.news_tree.yview)
        news_scroll.pack(side="right", fill="y")
        self.news_tree.configure(yscrollcommand=news_scroll.set)

        self.news_tree.tag_configure("up", foreground="#34d399")
        self.news_tree.tag_configure("down", foreground="#fca5a5")
        self.news_tree.tag_configure("neutral", foreground="#f8fafc")
        self.news_tree.tag_configure("newsCell", background="#111a33", foreground="#e2e8f0")

        footer = ttk.Frame(root, style="TFrame")
        footer.pack(fill="x", pady=(10, 0))
        ttk.Label(
            footer,
            text="※ 이 도구는 참고용 보조 지표이며 투자 조언이 아닙니다. 실제 매매 판단은 추가 확인 후 진행하세요.",
            style="Subtle.TLabel",
        ).pack(anchor="w")

    def _on_run(self) -> None:
        query = self.symbol_var.get().strip()
        if not query:
            messagebox.showwarning("입력 오류", "종목 또는 회사명을 입력해주세요.")
            return

        try:
            symbol = resolve_symbol_by_name(query)
        except Exception as exc:
            messagebox.showwarning("입력 오류", str(exc))
            return

        try:
            days = int(self.days_var.get())
            days = clamp(days, 1, 60)
        except ValueError:
            messagebox.showwarning("입력 오류", "예측 영업일은 1~60 정수여야 합니다.")
            return

        news_limit = 10

        period = self.period_var.get().strip()
        if period not in {"1mo", "3mo", "6mo", "1y", "2y"}:
            period = "6mo"

        self.status_var.set("현재 상태: 수집 중...")
        self.run_btn.state(["disabled"])
        self.symbol_var.set(symbol)
        self.days_var.set(str(days))

        for item in self.news_tree.get_children():
            self.news_tree.delete(item)
        self.news_records = []

        self.company_name_var.set("조회 중...")

        for val_label, sub_label in self._card_widgets.values():
            sub_label.config(text="업데이트 중...")
            val_label.config(text="-")

        def worker() -> None:
            try:
                result = build_analysis(symbol, days, news_limit, period)
                self.after(0, self._render_result, result)
            except Exception as exc:
                err_msg = str(exc)
                self.after(0, self._show_error, err_msg)

        threading.Thread(target=worker, daemon=True).start()

    def _render_result(self, result: Dict) -> None:
        company_name = (result.get("name") or "").strip()
        symbol = result.get("symbol", "").strip()
        self.company_name_var.set(company_name if company_name and company_name != symbol else symbol)
        self._set_card("current", f"{format_currency(result['current_price'], result['currency'])}", f"{result['name']} / {result['exchange']}")
        self._set_card("trend", f"{result['trend_pct']:+.2f}%", f"최근 추세 (기반: {result['horizon']}일)")
        if result.get("news_enabled"):
            sentiment_sub = f"{result['sentiment_label']} · 뉴스 {result['news_count']}건"
        else:
            sentiment_sub = "뉴스 수집 비활성화"
        self._set_card("sentiment", f"{result['news_sentiment']:+.2f}", sentiment_sub)
        self._set_card("forecast", f"{result['final_pct']:+.2f}%", f"신호: {result['signal']}")
        conf_sub = "과거 데이터" + (" + 뉴스 반영" if result.get("news_enabled") else "만 사용")
        self._set_card("confidence", f"{result['confidence']:.0f}%", conf_sub)
        self._set_card(
            "range",
            f"{format_currency(result['predicted'], result['currency'])}",
            f"{format_currency(result['lower'], result['currency'])} ~ {format_currency(result['upper'], result['currency'])}",
        )

        self.news_records = result["news_items"]
        for item in self.news_tree.get_children():
            self.news_tree.delete(item)

        for idx, row in enumerate(self.news_records):
            dt = row["published"]
            dt_text = dt.strftime("%Y-%m-%d %H:%M") if dt else "-"
            score = row.get("score", 0.0)
            tag = "neutral"
            if score >= 0.18:
                tag = "up"
            elif score <= -0.18:
                tag = "down"

            self.news_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(dt_text, row.get("provider", ""), f"{score:+.2f}", row.get("title", "")),
                tags=(tag, "newsCell"),
            )

        self._last_result = result
        self._draw_chart(
            result["dates"],
            result["prices"],
            result["candles"],
            predicted=result["predicted"],
            lower=result["lower"],
            upper=result["upper"],
            horizon=result["horizon"],
        )

        self.status_var.set(f"현재 상태: 완료 ({result['symbol']} / {result['latest_date']})")
        self.run_btn.state(["!disabled"])

    def _on_chart_resize(self, _event) -> None:
        if self._chart_resize_job is not None:
            self.after_cancel(self._chart_resize_job)
        self._chart_resize_job = self.after(80, self._redraw_chart)

    def _redraw_chart(self) -> None:
        self._chart_resize_job = None
        if self._last_result is None:
            return

        self._draw_chart(
            self._last_result["dates"],
            self._last_result["prices"],
            self._last_result["candles"],
            predicted=self._last_result["predicted"],
            lower=self._last_result["lower"],
            upper=self._last_result["upper"],
            horizon=self._last_result["horizon"],
        )

    def _show_error(self, msg: str) -> None:
        self.status_var.set("현재 상태: 실패")
        self.run_btn.state(["!disabled"])
        messagebox.showerror("오류", f"예측 데이터를 가져오지 못했습니다.\n{msg}")

    def _set_card(self, key: str, value: str, sub: str) -> None:
        if key not in self._card_widgets:
            return
        v_lbl, s_lbl = self._card_widgets[key]
        v_lbl.config(text=value)
        s_lbl.config(text=sub)

    def _draw_chart(
        self,
        dates: List[datetime],
        prices: List[float],
        candles: List[Tuple[float, float, float, float]] | None = None,
        predicted: float | None = None,
        lower: float | None = None,
        upper: float | None = None,
        horizon: int = 0,
    ) -> None:
        self.chart_canvas.delete("all")
        if not prices or len(prices) < 2:
            return

        w = max(10, self.chart_canvas.winfo_width())
        h = max(10, self.chart_canvas.winfo_height())
        pad_x_left = 24
        pad_x_right = 76
        pad_y = 22

        if w < 160:
            w = 560
        if h < 120:
            h = 260

        if not dates or len(dates) < len(prices):
            return

        plot_dates: List[datetime] = list(dates[-len(prices) :])
        plot_prices: List[float] = list(prices[-len(dates) :]) if len(prices) > len(dates) else list(prices)

        show_forecast = False
        if predicted is not None and plot_dates:
            future_days = max(1, min(int(horizon), 30))
            plot_dates.append(_add_trading_days(plot_dates[-1], future_days))
            plot_prices.append(float(predicted))
            show_forecast = True

        min_candidates: List[float] = list(plot_prices)
        if lower is not None:
            min_candidates.append(float(lower))
        if upper is not None:
            min_candidates.append(float(upper))

        min_p = min(min_candidates)
        max_p = max(min_candidates)
        if max_p == min_p:
            min_p -= 1
            max_p += 1
        span = max_p - min_p
        plot_len = len(plot_prices)
        denom = max(1, plot_len - 1)

        pts = []
        n = plot_len
        self._chart_points = []
        for i, p in enumerate(plot_prices):
            x = pad_x_left + (w - pad_x_left - pad_x_right) * (i / denom)
            y = (h - pad_y) - (p - min_p) / span * (h - pad_y * 2)
            pts.append((x, y))
            self._chart_points.append(
                (
                    i,
                    x,
                    plot_dates[i],
                    candles[i][0] if candles and i < len(candles) else p,  # open
                    candles[i][1] if candles and i < len(candles) else p,  # high
                    candles[i][2] if candles and i < len(candles) else p,  # low
                    candles[i][3] if candles and i < len(candles) else p,  # close
                )
            )
        self._chart_bounds = (pad_x_left, w - pad_x_right, pad_y, h - pad_y, min_p, max_p)

        # axis + grid (clear bottom/top indicators)
        self.chart_canvas.create_line(pad_x_left, h - pad_y, w - pad_x_right, h - pad_y, fill="#475569", width=2)
        self.chart_canvas.create_line(pad_x_left, pad_y, pad_x_left, h - pad_y, fill="#475569")
        for gi in range(1, 5):
            gy = pad_y + (h - pad_y * 2) * gi / 5
            value = max_p - (span * gi / 5)
            self.chart_canvas.create_line(pad_x_left, gy, w - pad_x_right, gy, fill="#1f2a44")
            self.chart_canvas.create_line(pad_x_left - 4, gy, pad_x_left, gy, fill="#475569")
            self.chart_canvas.create_text(
                pad_x_left - 6,
                gy,
                anchor="e",
                fill="#94a3b8",
                font=("Malgun Gothic", 8),
                text=f"{value:,.2f}",
            )

        x_ticks = 5
        for ti in range(x_ticks + 1):
            idx = int(round(ti * (len(plot_prices) - 1) / x_ticks)) if len(plot_prices) > 1 else 0
            idx = min(idx, len(plot_dates) - 1)
            x = pad_x_left + (w - pad_x_left - pad_x_right) * (idx / denom)
            self.chart_canvas.create_line(x, h - pad_y, x, h - pad_y + 4, fill="#334155")
            dt_text = plot_dates[idx].strftime("%m/%d")
            self.chart_canvas.create_text(
                x,
                h - pad_y + 10,
                text=dt_text,
                fill="#94a3b8",
                font=("Malgun Gothic", 8),
                anchor="n",
            )

        def price_to_y(value: float) -> float:
            return (h - pad_y) - (value - min_p) / span * (h - pad_y * 2)

        if candles:
            candle_w = max(3, int((w - pad_x_left - pad_x_right) / max(n - 1, 1) * 0.35))
            for i, c in enumerate(candles[: max(0, n - 1)]):
                open_price, high_price, low_price, close_price = c
                x = pts[i][0]

                y_open = price_to_y(open_price)
                y_close = price_to_y(close_price)
                y_high = price_to_y(high_price)
                y_low = price_to_y(low_price)

                color = "#ef4444"
                if close_price >= open_price:
                    color = "#22c55e"

                self.chart_canvas.create_line(x, y_high, x, y_low, fill="#94a3b8", width=1)
                y1 = min(y_open, y_close)
                y2 = max(y_open, y_close)
                if y2 - y1 < 1:
                    y2 = y1 + 1
                self.chart_canvas.create_rectangle(
                    x - candle_w / 2,
                    y1,
                    x + candle_w / 2,
                    y2,
                    fill=color,
                    outline=color,
                )
        else:
            for i in range(0, len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                self.chart_canvas.create_line(x1, y1, x2, y2, fill="#38bdf8", width=2.3)

        # Close price overlay
        for i in range(0, len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            self.chart_canvas.create_line(x1, y1, x2, y2, fill="#7dd3fc", width=1.4)

        if show_forecast and len(pts) >= 2:
            x1, y1 = pts[-2]
            x2, y2 = pts[-1]
            self.chart_canvas.create_line(x1, y1, x2, y2, fill="#f8fafc", width=2.2, dash=(7, 3))
            self.chart_canvas.create_oval(x2 - 4, y2 - 4, x2 + 4, y2 + 4, fill="#38bdf8", outline="#7dd3fc")
            forecast_text_x = min(w - pad_x_right + 6, x2 + 6)
            forecast_anchor = "w" if x2 + 6 <= w - pad_x_right + 6 else "e"
            self.chart_canvas.create_text(
                forecast_text_x,
                y2 - 10,
                anchor=forecast_anchor,
                text=f"예측({horizon}일): {predicted:,.2f}",
                fill="#93c5fd",
                font=("Malgun Gothic", 8),
            )

            if lower is not None and upper is not None:
                low_y = price_to_y(float(lower))
                high_y = price_to_y(float(upper))
                self.chart_canvas.create_line(x2, low_y, x2, high_y, fill="#64748b", width=1, dash=(3, 4))
                range_text_x = min(w - pad_x_right + 6, x2 + 6)
                range_anchor = "w" if x2 + 6 <= w - pad_x_right + 6 else "e"
                self.chart_canvas.create_text(
                    range_text_x,
                    y2 + 10,
                    anchor=range_anchor,
                    text=f"범위: {lower:,.2f} / {upper:,.2f}",
                    fill="#94a3b8",
                    font=("Malgun Gothic", 8),
                )

        self.chart_canvas.create_text(
            pad_x_left,
            pad_y - 10,
            anchor="w",
            text=f"최근 {len(prices)} 거래일 + 예측 {horizon}영업일" if show_forecast else f"최근 {len(prices)} 거래일",
            fill="#cbd5e1",
            font=("Malgun Gothic", 10, "bold"),
        )
        self.chart_canvas.create_line(pad_x_left, h - 42, w - pad_x_right, h - 42, fill="#334155")
        self.chart_canvas.create_text(
            pad_x_left,
            h - 34,
            anchor="w",
            text=f"최저: {min_p:,.2f}",
            fill="#94a3b8",
            font=("Malgun Gothic", 8),
        )
        self.chart_canvas.create_text(
            w - pad_x_right + 6,
            h - 34,
            anchor="w",
            text=f"최고: {max_p:,.2f}",
            fill="#94a3b8",
            font=("Malgun Gothic", 8),
        )

        self.chart_canvas.create_text(
            w / 2,
            h - 16,
            anchor="s",
            text="x축: 거래일",
            fill="#64748b",
            font=("Malgun Gothic", 8),
        )

    def _on_chart_hover(self, event) -> None:
        if not self._chart_points:
            return
        w = self.chart_canvas.winfo_width()
        h = self.chart_canvas.winfo_height()
        x_left, x_right, y_top, y_bottom, _, _ = self._chart_bounds
        if w <= 0 or h <= 0:
            return

        # Restrict hit-test to chart plot area only.
        if not (x_left <= event.x <= x_right and y_top <= event.y <= y_bottom):
            self._on_chart_leave()
            return

        idx = min(
            range(len(self._chart_points)),
            key=lambda i: abs(self._chart_points[i][1] - event.x),
        )
        _, _, dt, o, hi, lo, c = self._chart_points[idx]

        color = "#fca5a5" if c < o else "#34d399"
        change = (c - o) / o * 100 if o else 0.0
        tooltip = (
            f"{dt.strftime('%Y-%m-%d')}\n"
            f"Open: {o:,.2f}\n"
            f"High: {hi:,.2f}\n"
            f"Low: {lo:,.2f}\n"
            f"Close: {c:,.2f}\n"
            f"변동: {change:+.2f}%"
        )

        self.chart_tooltip.config(text=tooltip, fg=color)
        x = min(max(10, event.x + 12), max(0, w - 150))
        y = max(8, min(event.y - 80, max(0, h - 110)))
        self.chart_tooltip.place(x=x, y=y)

    def _on_chart_leave(self, _event=None) -> None:
        self.chart_tooltip.place_forget()

    def _open_news(self, _event) -> None:
        selection = self.news_tree.selection()
        if not selection:
            return
        idx = selection[0]
        try:
            record = self.news_records[int(idx)]
        except Exception:
            return

        link = (record.get("link") or "").strip()
        if not link:
            return
        webbrowser.open(link, new=2)


def main() -> None:
    app = StockNewsPredictorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
