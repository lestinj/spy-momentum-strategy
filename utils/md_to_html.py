#!/usr/bin/env python3
"""
Markdown to HTML Converter
Usage: python md_to_html.py <markdown_file.md>
"""

import sys
import os

def convert_md_to_html(md_file):
    """Convert markdown file to HTML and open in browser"""
    
    # Check if file exists
    if not os.path.exists(md_file):
        print(f"❌ Error: File '{md_file}' not found")
        return
    
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Try using markdown library (preferred)
    try:
        import markdown
        html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite'])
    except ImportError:
        # Fallback: basic conversion
        print("⚠️  'markdown' library not found. Using basic conversion.")
        print("   For better results, install it: pip install markdown")
        html_body = f"<pre>{md_content}</pre>"
    
    # Create full HTML with styling
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{os.path.basename(md_file)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
        }}
        h1, h2, h3, h4, h5, h6 {{
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }}
        h1 {{ font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
        h2 {{ font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
        h3 {{ font-size: 1.25em; }}
        code {{
            background: #f6f8fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 85%;
        }}
        pre {{
            background: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow: auto;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }}
        th, td {{
            border: 1px solid #dfe2e5;
            padding: 8px 13px;
            text-align: left;
        }}
        th {{
            background: #f6f8fa;
            font-weight: 600;
        }}
        blockquote {{
            border-left: 4px solid #dfe2e5;
            padding-left: 16px;
            color: #6a737d;
            margin: 16px 0;
        }}
        a {{
            color: #0366d6;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        hr {{
            border: 0;
            border-top: 1px solid #eaecef;
            margin: 24px 0;
        }}
        ul, ol {{
            padding-left: 2em;
            margin: 16px 0;
        }}
        li {{
            margin: 4px 0;
        }}
        .header {{
            background: #0366d6;
            color: white;
            padding: 20px;
            margin: -40px -20px 40px -20px;
            border-radius: 6px 6px 0 0;
        }}
        .footer {{
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid #eaecef;
            color: #6a737d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0; border: none; padding: 0;">📄 {os.path.basename(md_file)}</h1>
    </div>
    {html_body}
    <div class="footer">
        <p>Converted from: <code>{os.path.abspath(md_file)}</code></p>
    </div>
</body>
</html>"""
    
    # Create output HTML file
    html_file = md_file.rsplit('.', 1)[0] + '.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"✅ Converted successfully!")
    print(f"   Input:  {md_file}")
    print(f"   Output: {html_file}")
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(html_file))
        print(f"🌐 Opening in browser...")
    except:
        print(f"\n💡 Open this file in your browser:")
        print(f"   {os.path.abspath(html_file)}")
    
    return html_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_html.py <markdown_file.md>")
        print("\nExample:")
        print("  python md_to_html.py YTD_2025_TESTING_GUIDE.md")
        sys.exit(1)
    
    md_file = sys.argv[1]
    convert_md_to_html(md_file)
