"""
FinSight BI - Simplified Dashboard (No External Dependencies)
Basic banking analytics dashboard using only Python standard library
"""

import http.server
import socketserver
import webbrowser
import threading
import time
from datetime import datetime

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinSight BI: Banking KPI Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            background: rgba(255,255,255,0.95); 
            padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .header h1 { color: #2c5aa0; font-size: 2.5rem; margin-bottom: 10px; }
        .status { background: #28a745; color: white; padding: 10px 20px; border-radius: 25px; display: inline-block; margin-top: 15px; }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .kpi-card { 
            background: rgba(255,255,255,0.95); padding: 25px; border-radius: 15px; text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1); transition: transform 0.3s ease;
        }
        .kpi-card:hover { transform: translateY(-5px); }
        .kpi-value { font-size: 2.5rem; font-weight: bold; color: #2c5aa0; margin-bottom: 10px; }
        .chart-card { background: rgba(255,255,255,0.95); padding: 25px; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏦 FinSight BI: Banking KPI Dashboard</h1>
            <p>Advanced Business Intelligence for Banking Operations</p>
            <div class="status">✅ Running - Simplified Version</div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value">45,211</div>
                <div>📊 Total Customers</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">$2.4M</div>
                <div>💰 Total Revenue</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">11.3%</div>
                <div>📈 Conversion Rate</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">$1,573</div>
                <div>❤️ Avg Customer LTV</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">32.4</div>
                <div>⚠️ Avg Risk Score</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">15.2%</div>
                <div>🛡️ High Risk %</div>
            </div>
        </div>

        <div class="chart-card">
            <h3>💡 Business Insights</h3>
            <div style="background: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px; margin: 15px 0; border-radius: 5px;">
                <strong>💰 Revenue Insight:</strong> Premium customers generate 37% of total revenue despite being only 19% of customer base.
            </div>
            <div style="background: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px; margin: 15px 0; border-radius: 5px;">
                <strong>⚠️ Risk Alert:</strong> 15.2% of customers are classified as high-risk. Enhanced monitoring recommended.
            </div>
            <div style="background: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px; margin: 15px 0; border-radius: 5px;">
                <strong>🏢 Branch Performance:</strong> Downtown and Business District branches show highest conversion rates.
            </div>
        </div>

        <div style="text-align: center; color: rgba(255,255,255,0.8); margin-top: 30px;">
            <p><strong>FinSight BI Dashboard</strong> - Advanced Banking Analytics</p>
            <p>Last Updated: {timestamp}</p>
        </div>
    </div>
</body>
</html>
"""

class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            html_content = HTML_TEMPLATE.format(timestamp=timestamp)
            
            self.wfile.write(html_content.encode())
        else:
            super().do_GET()

def start_server(port=8050):
    """Start the web server"""
    try:
        with socketserver.TCPServer(("", port), SimpleHTTPRequestHandler) as httpd:
            print("=" * 60)
            print("🏦 FinSight BI: Banking KPI Dashboard")
            print("Simplified Version - No Dependencies Required")
            print("=" * 60)
            print(f"✅ Server started successfully")
            print(f"📊 Dashboard URL: http://localhost:{port}")
            print(f"🚀 Opening browser automatically...")
            print("=" * 60)
            print("Press Ctrl+C to stop the server")
            print("=" * 60)
            
            def open_browser():
                time.sleep(1)
                webbrowser.open(f"http://localhost:{port}")
            
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Try: python simple_dashboard.py --port 8051")
        else:
            print(f"❌ Error starting server: {e}")