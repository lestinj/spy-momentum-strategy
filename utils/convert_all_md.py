#!/usr/bin/env python3
"""
Batch Markdown to HTML Converter
Converts all .md files in current directory to HTML
"""

import os
import glob

def batch_convert():
    """Convert all markdown files in current directory"""
    
    # Find all .md files
    md_files = glob.glob("*.md")
    
    if not md_files:
        print("❌ No markdown files found in current directory")
        return
    
    print(f"📚 Found {len(md_files)} markdown file(s)\n")
    
    # Try to import markdown library
    try:
        import markdown
        has_markdown = True
    except ImportError:
        print("⚠️  'markdown' library not found. Using basic conversion.")
        print("   For better results: pip install markdown\n")
        has_markdown = False
    
    converted = []
    
    for md_file in md_files:
        print(f"Converting: {md_file}...", end=" ")
        
        try:
            # Read markdown
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert to HTML
            if has_markdown:
                html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite'])
            else:
                html_body = f"<pre>{md_content}</pre>"
            
            # Create styled HTML
            html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{md_file}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
        }}
        h1, h2, h3 {{ 
            margin-top: 24px; 
            font-weight: 600; 
        }}
        h1 {{ font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
        h2 {{ font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
        code {{
            background: #f6f8fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }}
        pre {{
            background: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow: auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }}
        th, td {{
            border: 1px solid #dfe2e5;
            padding: 8px 13px;
        }}
        th {{ background: #f6f8fa; font-weight: 600; }}
        .header {{
            background: #0366d6;
            color: white;
            padding: 20px;
            margin: -40px -20px 40px -20px;
            border-radius: 6px 6px 0 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0; border: none;">📄 {md_file}</h1>
    </div>
    {html_body}
</body>
</html>"""
            
            # Write HTML file
            html_file = md_file.rsplit('.', 1)[0] + '.html'
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html)
            
            converted.append(html_file)
            print("✅")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Converted {len(converted)} file(s):")
    for html_file in converted:
        print(f"   • {html_file}")
    print(f"{'='*60}")
    print("\n💡 Open any .html file in your browser to view!")

if __name__ == "__main__":
    batch_convert()
