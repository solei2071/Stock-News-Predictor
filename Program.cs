using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Web.Script.Serialization;

namespace StockNewsPredictor
{
    class NewsItem
    {
        public string Title { get; set; }
        public string Link { get; set; }
        public string Summary { get; set; }
        public string Provider { get; set; }
        public DateTime? Published { get; set; }
        public double Sentiment { get; set; }
    }

    class PriceInfo
    {
        public string Symbol { get; set; }
        public string Name { get; set; }
        public string Currency { get; set; }
        public string Exchange { get; set; }
        public double CurrentPrice { get; set; }
        public DateTime LastDate { get; set; }
        public List<double> Prices { get; set; }
    }

    class Program
    {
        static readonly Dictionary<string, int> Positive = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
        {
            {"beat", 2}, {"surge", 2}, {"surged", 2}, {"raise", 1}, {"raised", 1},
            {"rally", 2}, {"rallies", 2}, {"growth", 2}, {"record", 1}, {"upgrade", 2},
            {"upside", 1}, {"profit", 1}, {"strong", 1}, {"expansion", 1}, {"improve", 1},
            {"improved", 1}, {"success", 1}, {"buy", 1}, {"bullish", 2}, {"innovation", 1},
            {"혁신", 2}, {"성장", 2}, {"호재", 2}, {"상승", 2}, {"개선", 1}, {"흑자", 1}
        };

        static readonly Dictionary<string, int> Negative = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
        {
            {"miss", 2}, {"missed", 2}, {"drop", 2}, {"dropped", 2}, {"fall", 2}, {"fell", 2},
            {"loss", 2}, {"losses", 2}, {"downgrade", 2}, {"warning", 1}, {"weak", 1}, {"weakness", 1},
            {"risk", 1}, {"lawsuit", 2}, {"investigation", 1}, {"inflation", 1}, {"recession", 2},
            {"decline", 2}, {"downside", 1}, {"bearish", 2}, {"부진", 2}, {"감소", 2}, {"적자", 2}, {"우려", 1}
        };

        const string USER_AGENT = "Mozilla/5.0 (compatible; StockNewsPredictor/1.0)";

        static int Main(string[] args)
        {
            try
            {
                if (args == null || args.Length == 0)
                {
                    Console.WriteLine("사용법: StockNewsPredictor.exe [티커] [--days n] [--news n] [--period 6mo]");
                    Console.WriteLine("예: StockNewsPredictor.exe AAPL --days 7 --news 20 --period 6mo");
                    return 1;
                }

                string symbol = args[0].Trim().ToUpperInvariant();
                int days = 7;
                int newsCount = 20;
                string period = "6mo";

                for (int i = 1; i < args.Length; i++)
                {
                    if (args[i].Equals("--days", StringComparison.OrdinalIgnoreCase) || args[i].Equals("/days", StringComparison.OrdinalIgnoreCase))
                    {
                        if (i + 1 < args.Length) { days = ParsePositiveInt(args[i + 1], 7); i++; }
                    }
                    else if (args[i].Equals("--news", StringComparison.OrdinalIgnoreCase) || args[i].Equals("/news", StringComparison.OrdinalIgnoreCase))
                    {
                        if (i + 1 < args.Length) { newsCount = ParsePositiveInt(args[i + 1], 20); i++; }
                    }
                    else if (args[i].Equals("--period", StringComparison.OrdinalIgnoreCase) || args[i].Equals("/period", StringComparison.OrdinalIgnoreCase))
                    {
                        if (i + 1 < args.Length) { period = args[i + 1]; i++; }
                    }
                    else if (args[i].Contains(":"))
                    {
                        // slash syntax: /days:7, /news:20, /period:6mo
                        var sep = args[i].Split(':');
                        if (sep.Length == 2)
                        {
                            if (sep[0].Equals("/days", StringComparison.OrdinalIgnoreCase)) days = ParsePositiveInt(sep[1], 7);
                            else if (sep[0].Equals("/news", StringComparison.OrdinalIgnoreCase)) newsCount = ParsePositiveInt(sep[1], 20);
                            else if (sep[0].Equals("/period", StringComparison.OrdinalIgnoreCase)) period = sep[1];
                        }
                    }
                }

                days = Math.Max(1, Math.Min(60, days));
                newsCount = Math.Max(1, newsCount);

                var report = BuildReport(symbol, days, newsCount, period);
                Console.WriteLine(report);
                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine("오류: " + ex.Message);
                return 1;
            }
        }

        static int ParsePositiveInt(string value, int fallback)
        {
            if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) && n > 0)
                return n;
            return fallback;
        }

        static string FetchJson(string url)
        {
            using (var wc = new WebClient())
            {
                wc.Headers[HttpRequestHeader.UserAgent] = USER_AGENT;
                wc.Headers[HttpRequestHeader.Accept] = "application/json";
                return wc.DownloadString(url);
            }
        }

        static Dictionary<string, object> ParseObject(string json)
        {
            var serializer = new JavaScriptSerializer();
            serializer.MaxJsonLength = int.MaxValue;
            return serializer.Deserialize<Dictionary<string, object>>(json);
        }

        static PriceInfo ParsePriceData(string symbol, string period)
        {
            string url = string.Format(CultureInfo.InvariantCulture,
                "https://query1.finance.yahoo.com/v8/finance/chart/{0}?interval=1d&range={1}&includeAdjustedClose=true",
                Uri.EscapeDataString(symbol), Uri.EscapeDataString(period));
            var json = FetchJson(url);
            var chart = GetDict(json, "chart");
            var result = GetList(chart, "result");
            if (result == null || result.Length == 0)
                throw new Exception("주가 데이터가 없습니다.");

            var first = result[0] as Dictionary<string, object>;
            if (first == null)
                throw new Exception("가격 데이터 형식이 올바르지 않습니다.");

            var timestamps = GetList(first, "timestamp");
            var indicators = GetDict(first, "indicators");
            var quoteList = GetList(indicators, "quote");
            if (quoteList == null || quoteList.Length == 0)
                throw new Exception("가격 지표가 없습니다.");

            var closeList = GetList(quoteList[0] as Dictionary<string, object>, "close");
            if (timestamps == null || closeList == null || timestamps.Length != closeList.Length)
                throw new Exception("종가 데이터가 비어 있습니다.");

            var prices = new List<double>();
            DateTime lastDate = DateTime.UtcNow;
            for (int i = 0; i < timestamps.Length; i++)
            {
                if (closeList[i] == null)
                    continue;
                double close = ToDouble(closeList[i]);
                if (double.IsNaN(close) || double.IsInfinity(close))
                    continue;

                long ts = ToLong(timestamps[i]);
                if (ts <= 0) continue;
                DateTime dt = DateTimeOffset.FromUnixTimeSeconds(ts).UtcDateTime;
                prices.Add(close);
                lastDate = dt;
            }

            if (prices.Count == 0)
                throw new Exception("유효한 가격 데이터가 없습니다.");

            var meta = GetDict(first, "meta");
            return new PriceInfo
            {
                Symbol = GetString(meta, "symbol", symbol),
                Name = GetString(meta, "longName", GetString(meta, "shortName", symbol)),
                Currency = GetString(meta, "currency", "USD"),
                Exchange = GetString(meta, "exchangeName", "N/A"),
                CurrentPrice = GetDouble(meta, "regularMarketPrice", prices[prices.Count - 1]),
                LastDate = lastDate,
                Prices = prices
            };
        }

        static IEnumerable<Dictionary<string, object>> FindNewsNodes(Dictionary<string, object> root)
        {
            var collected = new List<Dictionary<string, object>>();

            if (root == null) return collected;

            if (root.TryGetValue("finance", out object finObj) && finObj is Dictionary<string, object> finDict)
            {
                var finResult = GetList(finDict, "result");
                if (finResult != null)
                {
                    foreach (var item in finResult)
                    {
                        var d = item as Dictionary<string, object>;
                        if (d == null) continue;
                        var news = GetList(d, "news");
                        if (news != null)
                        {
                            collected.AddRange(news.OfType<Dictionary<string, object>>());
                        }
                    }
                }
            }

            if (root.TryGetValue("news", out object directNews) && directNews is object[] directList)
            {
                collected.AddRange(directList.OfType<Dictionary<string, object>>());
            }

            // fallback recursive search for any nested `news` lists
            RecurseForNews(root, collected);
            return collected;
        }

        static void RecurseForNews(object node, List<Dictionary<string, object>> sink)
        {
            if (node is Dictionary<string, object> dict)
            {
                foreach (var kv in dict)
                {
                    if (kv.Key != null && kv.Key.Equals("news", StringComparison.OrdinalIgnoreCase)
                        && kv.Value is object[] arr)
                    {
                        sink.AddRange(arr.OfType<Dictionary<string, object>>());
                        continue;
                    }
                    RecurseForNews(kv.Value, sink);
                }
            }
            else if (node is IEnumerable arr && !(node is string))
            {
                foreach (var it in arr)
                {
                    RecurseForNews(it, sink);
                }
            }
        }

        static List<NewsItem> FetchNews(string symbol, int limit)
        {
            string url = string.Format(CultureInfo.InvariantCulture,
                "https://query1.finance.yahoo.com/v1/finance/search?q={0}&quotesCount=0&newsCount={1}&enableFuzzyQuery=false",
                Uri.EscapeDataString(symbol), limit);
            var json = FetchJson(url);
            var payload = ParseObject(json);
            var raw = FindNewsNodes(payload).ToList();

            // remove duplicates
            var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var result = new List<NewsItem>();
            foreach (var n in raw)
            {
                string title = GetString(n, "title", string.Empty).Trim();
                if (string.IsNullOrEmpty(title))
                    continue;

                string link = GetString(n, "link", string.Empty);
                if (string.IsNullOrWhiteSpace(link))
                    link = GetString(n, "url", string.Empty);
                if (string.IsNullOrWhiteSpace(link))
                    link = GetString(n, "canonicalUrl", string.Empty);
                if (string.IsNullOrWhiteSpace(link))
                    link = GetString(n, "linkUrl", string.Empty);

                string key = string.IsNullOrWhiteSpace(link) ? title : link;
                if (!set.Add(key))
                    continue;

                var p = ParseNewsTime(
                    GetAny(n, "providerPublishTime") ??
                    GetAny(n, "pubDate") ??
                    GetAny(n, "date") ??
                    GetAny(n, "published")
                );

                var provider = GetAny(n, "publisher") ?? GetAny(n, "provider") ?? GetAny(n, "source");
                if (provider is Dictionary<string, object> provDict && provDict.TryGetValue("name", out object provName))
                    provider = provName;

                result.Add(new NewsItem
                {
                    Title = title,
                    Link = link,
                    Summary = GetString(n, "summary", GetString(n, "excerpt", GetString(n, "description", string.Empty)),
                    Provider = provider?.ToString() ?? "unknown",
                    Published = p
                });
            }

            return result
                .OrderByDescending(x => x.Published ?? DateTime.MinValue)
                .Take(limit)
                .ToList();
        }

        static DateTime? ParseNewsTime(object raw)
        {
            if (raw == null)
                return null;

            if (raw is int i)
            {
                long ms = i;
                if (ms > 1_000_000_000_000L)
                    ms /= 1000;
                return DateTimeOffset.FromUnixTimeSeconds(ms).UtcDateTime;
            }

            if (raw is long l)
            {
                long ms = l;
                if (ms > 1_000_000_000_000L)
                    ms /= 1000;
                return DateTimeOffset.FromUnixTimeSeconds(ms).UtcDateTime;
            }

            if (raw is double db)
            {
                long ms = (long)db;
                if (ms > 1_000_000_000_000L)
                    ms /= 1000;
                return DateTimeOffset.FromUnixTimeSeconds(ms).UtcDateTime;
            }

            if (raw is string s)
            {
                if (long.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out long secs))
                {
                    if (secs > 1_000_000_000_000L)
                        secs /= 1000;
                    return DateTimeOffset.FromUnixTimeSeconds(secs).UtcDateTime;
                }
                if (DateTimeOffset.TryParse(s, out var dt))
                    return dt.UtcDateTime;
            }

            return null;
        }

        static Tuple<double, List<Tuple<string, double>>> SentimentScore(List<NewsItem> items)
        {
            if (items == null || items.Count == 0)
                return Tuple.Create(0.0, new List<Tuple<string, double>>());

            double total = 0.0;
            var perItem = new List<Tuple<string, double>>();

            foreach (var item in items)
            {
                string text = (item.Title + " " + item.Summary).ToLowerInvariant();
                var tokens = Regex.Matches(text, "[A-Za-z가-힣]+");
                if (tokens.Count == 0)
                {
                    perItem.Add(Tuple.Create(item.Title, 0.0));
                    continue;
                }

                double score = 0.0;
                foreach (Match m in tokens)
                {
                    string t = m.Value;
                    if (Positive.TryGetValue(t, out int p)) score += p;
                    if (Negative.TryGetValue(t, out int n)) score -= n;
                }
                score /= Math.Max(tokens.Count, 1);
                item.Sentiment = score;
                total += score;
                perItem.Add(Tuple.Create(item.Title, score));
            }

            return Tuple.Create(Math.Max(-1.0, Math.Min(1.0, total / items.Count)), perItem);
        }

        static double LinearForecast(List<double> prices, int horizon)
        {
            int n = prices.Count;
            if (n < 3 || horizon <= 0)
                return 0.0;

            var ys = prices.Select(Math.Log).ToArray();
            double[] xs = Enumerable.Range(0, n).Select(x => (double)x).ToArray();

            double meanX = xs.Average();
            double meanY = ys.Average();

            double denom = 0.0;
            double num = 0.0;
            for (int i = 0; i < n; i++)
            {
                double dx = xs[i] - meanX;
                denom += dx * dx;
                num += dx * (ys[i] - meanY);
            }

            if (Math.Abs(denom) < 1e-12)
                return 0.0;

            double slope = num / denom;
            double intercept = meanY - slope * meanX;
            double lastX = xs[n - 1];
            double forecastX = lastX + horizon;
            double forecastLog = intercept + slope * forecastX;
            double trend = (Math.Exp(forecastLog - Math.Log(prices[n - 1])) - 1.0) * 100.0;
            return trend;
        }

        static double Volatility(List<double> prices)
        {
            if (prices.Count < 2)
                return 0.25;
            var rets = new List<double>();
            for (int i = 1; i < prices.Count; i++)
            {
                double prev = prices[i - 1];
                double curr = prices[i];
                if (prev <= 0)
                    continue;
                rets.Add(Math.Log(curr / prev));
            }

            if (rets.Count < 2)
                return 0.25;

            double avg = rets.Average();
            double sum = 0.0;
            foreach (var r in rets) sum += (r - avg) * (r - avg);
            return Math.Sqrt(sum / rets.Count);
        }

        static string FormatCurrency(double v, string currency)
        {
            if (string.Equals(currency, "USD", StringComparison.OrdinalIgnoreCase))
                return string.Format(CultureInfo.InvariantCulture, "${0:N2}", v);
            return string.Format(CultureInfo.InvariantCulture, "{0:N2} {1}", v, currency);
        }

        static string BuildReport(string symbol, int horizon, int newsCount, string period)
        {
            var price = ParsePriceData(symbol, period);
            var news = FetchNews(symbol, newsLimit: newsCount);
            var sentimentResult = SentimentScore(news);
            double sentiment = sentimentResult.Item1;
            var perItem = sentimentResult.Item2;

            double trendPct = LinearForecast(price.Prices, horizon);
            double sentimentAdj = sentiment * Math.Min(Math.Max(horizon, 1) / 7.0, 2.0) * 2.5;
            double finalPct = trendPct + sentimentAdj;

            double vol = Volatility(price.Prices);
            double band = Clamp(vol * 100.0 * Math.Sqrt(horizon), 1.5, 12.0);

            double lastPrice = price.Prices[price.Prices.Count - 1];
            double predicted = lastPrice * (1 + finalPct / 100.0);
            double lower = lastPrice * (1 + (finalPct - band) / 100.0);
            double upper = lastPrice * (1 + (finalPct + band) / 100.0);

            double dataConf = Clamp(price.Prices.Count / 120.0 * 100.0, 0, 100);
            double newsConf = Clamp(news.Count / 20.0 * 100.0, 0, 100);
            double trendConf = Clamp(70 - (Math.Abs(vol) * 120), 20, 75);
            double confidence = Clamp(15 + 0.50 * dataConf + 0.25 * newsConf + 0.25 * trendConf, 5, 95);

            var sb = new StringBuilder();
            sb.AppendLine();
            sb.AppendLine($"[예측 결과] {price.Symbol} ({price.Name})");
            sb.AppendLine($"거래소: {price.Exchange} | 통화: {price.Currency}");
            sb.AppendLine($"최근 거래일: {price.LastDate:yyyy-MM-dd} | 현재가: {FormatCurrency(price.CurrentPrice, price.Currency)}");
            sb.AppendLine($"예측 기간: {horizon} 거래일");
            sb.AppendLine();
            sb.AppendLine($"가격 트렌드 추정 수익률: {trendPct:+0.00;-0.00;0.00}%");
            sb.AppendLine($"뉴스 감성 점수: {sentiment:+0.000;-0.000;0.000} (-1.000 ~ 1.000)");
            sb.AppendLine($"예측 수익률(감성 반영): {finalPct:+0.00;-0.00;0.00}%");
            sb.AppendLine($"예상 종가: {FormatCurrency(predicted, price.Currency)}");
            sb.AppendLine($"예상 구간: {FormatCurrency(lower, price.Currency)} ~ {FormatCurrency(upper, price.Currency)}");
            sb.AppendLine($"모형 신뢰도: {confidence:0}");
            sb.AppendLine();
            sb.AppendLine($"수집 뉴스 ({news.Count}건, 최근순):");

            foreach (var item in news.Take(Math.Min(5, news.Count)))
            {
                string dateText = item.Published.HasValue ? item.Published.Value.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture) : "날짜없음";
                sb.AppendLine($"- {dateText} | {item.Provider} | {item.Sentiment:+0.000;-0.000;0.000} | {Truncate(item.Title, 90)}");
            }

            sb.AppendLine();
            sb.AppendLine("※ 이 도구는 참고용 예측이며 투자 조언이 아닙니다.");
            return sb.ToString();
        }

        static string Truncate(string text, int len)
        {
            if (string.IsNullOrEmpty(text)) return string.Empty;
            if (text.Length <= len) return text;
            return text.Substring(0, Math.Max(0, len - 3)) + "...";
        }

        static double Clamp(double v, double min, double max) => Math.Max(min, Math.Min(max, v));

        static Dictionary<string, object> GetDict(Dictionary<string, object> node, string key)
        {
            if (node == null) return null;
            if (node.TryGetValue(key, out object value) && value is Dictionary<string, object> dict)
                return dict;
            return null;
        }

        static object[] GetList(Dictionary<string, object> node, string key)
        {
            if (node == null) return null;
            if (node.TryGetValue(key, out object value) && value is object[] arr)
                return arr;
            return null;
        }

        static string GetString(Dictionary<string, object> node, string key, string fallback = "")
        {
            if (node != null && node.TryGetValue(key, out object value) && value != null)
                return Convert.ToString(value, CultureInfo.InvariantCulture);
            return fallback;
        }

        static double GetDouble(Dictionary<string, object> node, string key, double fallback = 0.0)
        {
            if (node != null && node.TryGetValue(key, out object value))
                return ToDouble(value);
            return fallback;
        }

        static object GetAny(Dictionary<string, object> node, string key)
        {
            if (node != null && node.TryGetValue(key, out object value))
                return value;
            return null;
        }

        static long ToLong(object value)
        {
            if (value == null)
                return 0L;
            if (value is int i) return i;
            if (value is long l) return l;
            if (value is double d) return Convert.ToInt64(d);
            if (value is decimal m) return Convert.ToInt64(m);
            long parsed;
            return long.TryParse(Convert.ToString(value, CultureInfo.InvariantCulture), NumberStyles.Integer, CultureInfo.InvariantCulture, out parsed) ? parsed : 0L;
        }

        static double ToDouble(object value)
        {
            if (value == null) return double.NaN;
            if (value is double d) return d;
            if (value is float f) return f;
            if (value is int i) return i;
            if (value is long l) return l;
            if (value is decimal m) return (double)m;
            if (double.TryParse(Convert.ToString(value, CultureInfo.InvariantCulture), NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed))
                return parsed;
            return double.NaN;
        }
    }
}
