"""Web tools cho AI-local agent: fetch URL và tìm kiếm DuckDuckGo."""

import json
import re
import urllib.parse
import urllib.request
import urllib.error


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AI-local-agent/1.0)",
    "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi,en;q=0.9",
}
_MAX_CONTENT = 50_000  # ký tự


def web_fetch(url: str, extract_text: bool = True) -> str:
    """
    Lấy nội dung từ URL.
    - Với HTML: trích xuất text thuần (bỏ tags).
    - Với JSON: trả về JSON đẹp.
    - Với text: trả về nguyên.
    """
    if not url.startswith(("http://", "https://")):
        return f"[ERROR] URL phải bắt đầu bằng http:// hoặc https://"

    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()

        # Decode
        encoding = "utf-8"
        if "charset=" in content_type:
            encoding = content_type.split("charset=")[-1].split(";")[0].strip()
        try:
            text = raw.decode(encoding, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")

        # JSON
        if "json" in content_type or text.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(text)
                return json.dumps(parsed, ensure_ascii=False, indent=2)[:_MAX_CONTENT]
            except json.JSONDecodeError:
                pass

        # HTML → text thuần
        if extract_text and ("html" in content_type or "<html" in text.lower()[:200]):
            text = _html_to_text(text)

        return text[:_MAX_CONTENT]

    except urllib.error.HTTPError as e:
        return f"[ERROR] HTTP {e.code}: {e.reason} — {url}"
    except urllib.error.URLError as e:
        return f"[ERROR] Không truy cập được URL: {e.reason}"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


def web_search(query: str, max_results: int = 5) -> str:
    """
    Tìm kiếm web qua DuckDuckGo HTML (không cần API key).
    Trả về danh sách kết quả với tiêu đề, URL, mô tả.
    """
    if not query.strip():
        return "[ERROR] Từ khóa tìm kiếm trống."

    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"

    try:
        req = urllib.request.Request(url, headers={**_HEADERS, "Accept": "text/html"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"[ERROR] Không tìm kiếm được: {e}"

    # Parse kết quả từ HTML DuckDuckGo
    results = []
    # Tìm các result block
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )

    if not blocks:
        # Fallback: tìm link đơn giản
        links = re.findall(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
        for i, (href, title) in enumerate(links[:max_results]):
            title_clean = _strip_tags(title).strip()
            # DuckDuckGo dùng redirect URL
            actual_url = _extract_ddg_url(href)
            results.append(f"{i+1}. **{title_clean}**\n   URL: {actual_url}")
    else:
        for i, (href, title, snippet) in enumerate(blocks[:max_results]):
            title_clean = _strip_tags(title).strip()
            snippet_clean = _strip_tags(snippet).strip()
            actual_url = _extract_ddg_url(href)
            results.append(f"{i+1}. **{title_clean}**\n   URL: {actual_url}\n   {snippet_clean}")

    if not results:
        return f"Không tìm thấy kết quả cho: '{query}'"
    return f"# Kết quả tìm kiếm: '{query}'\n\n" + "\n\n".join(results)


def _html_to_text(html: str) -> str:
    """Chuyển HTML thành text thuần, giữ cấu trúc cơ bản."""
    # Bỏ script/style
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Header tags → tiêu đề
    html = re.sub(r"<h[1-6][^>]*>", "\n## ", html, flags=re.IGNORECASE)
    html = re.sub(r"</h[1-6]>", "\n", html, flags=re.IGNORECASE)
    # Paragraph/div/br → xuống dòng
    html = re.sub(r"<(p|div|br|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Bỏ tất cả tags còn lại
    html = re.sub(r"<[^>]+>", "", html)
    # Decode HTML entities
    html = html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    html = html.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Chuẩn hóa khoảng trắng
    lines = [ln.strip() for ln in html.splitlines()]
    lines = [ln for ln in lines if ln]
    # Loại bỏ dòng lặp
    deduped = []
    for ln in lines:
        if not deduped or ln != deduped[-1]:
            deduped.append(ln)
    return "\n".join(deduped)


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def _extract_ddg_url(href: str) -> str:
    """DuckDuckGo encode URL trong redirect, trích xuất URL thật."""
    if href.startswith("//duckduckgo.com/l/?"):
        params = urllib.parse.parse_qs(urllib.parse.urlparse("https:" + href).query)
        uddg = params.get("uddg", [""])[0]
        if uddg:
            return urllib.parse.unquote(uddg)
    if href.startswith("/"):
        return "https://duckduckgo.com" + href
    return href
