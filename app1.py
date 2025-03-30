import requests
from flask import Flask, request, jsonify
from datetime import datetime
from bs4 import BeautifulSoup
import json
import os

app = Flask(__name__)

# Store monitored sites in a dictionary
monitored_sites = {}

class WebScraper:
    def __init__(self, use_tor=False):
        self.session = requests.Session()
        if use_tor:
            # Configure for Tor if requested
            self.session.proxies.update({
                'http': 'socks5h://127.0.0.1:9150',  # Use port 9150 for Tor Browser
                'https': 'socks5h://127.0.0.1:9150'
            })
            print("Tor proxy configured:", self.session.proxies)  # Debugging
    
    def fetch_website_content(self, url, timeout=30):  # Increased timeout
        """Fetch content from a website"""
        try:
            print(f"Fetching URL: {url}")  # Debugging
            print(f"Using Tor: {self.session.proxies}")  # Debugging
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_website_info(self, html_content):
        """Extract detailed information from website HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract basic info
        title = soup.title.string if soup.title else 'No Title'
        meta_description = soup.find('meta', attrs={'name': 'description'})['content'] \
            if soup.find('meta', attrs={'name': 'description'}) else 'No Description'
        links = [a['href'] for a in soup.find_all('a', href=True)]
        images = [img['src'] for img in soup.find_all('img', src=True)]
        text_length = len(soup.get_text())
        
        # Identify website type
        website_type = self.identify_website_type(soup)
        
        # Extract payment methods (if applicable)
        payment_methods = self.extract_payment_methods(soup)
        
        # Extract transactions (if applicable)
        transactions = self.extract_transactions(soup)
        
        info = {
            'title': title,
            'meta_description': meta_description,
            'links': links,
            'images': images,
            'text_length': text_length,
            'website_type': website_type,
            'payment_methods': payment_methods,
            'transactions': transactions
        }
        return info
    
    def identify_website_type(self, soup):
        """Identify the type of website based on content and meta tags"""
        # Common keywords for different types of websites
        ecommerce_keywords = ['shop', 'cart', 'buy', 'product', 'store']
        blog_keywords = ['blog', 'post', 'article', 'comment']
        news_keywords = ['news', 'headline', 'breaking', 'article']
        social_media_keywords = ['login', 'signup', 'profile', 'share']
        payment_keywords = ['payment', 'checkout', 'credit card', 'paypal']
        
        # Get all text content
        text_content = soup.get_text().lower()
        
        # Check for e-commerce
        if any(keyword in text_content for keyword in ecommerce_keywords):
            return 'E-commerce'
        
        # Check for blog
        if any(keyword in text_content for keyword in blog_keywords):
            return 'Blog'
        
        # Check for news
        if any(keyword in text_content for keyword in news_keywords):
            return 'News'
        
        # Check for social media
        if any(keyword in text_content for keyword in social_media_keywords):
            return 'Social Media'
        
        # Check for payment-related sites
        if any(keyword in text_content for keyword in payment_keywords):
            return 'Payment Gateway'
        
        # Default to 'Unknown' if no type is identified
        return 'Unknown'
    
    def extract_payment_methods(self, soup):
        """Extract payment methods from the website (if applicable)"""
        # Example: Look for payment-related elements
        payment_methods = []
        for element in soup.find_all(class_='payment-method'):  # Adjust class name as needed
            payment_methods.append(element.get_text())
        return payment_methods
    
    def extract_transactions(self, soup):
        """Extract transactions from the website (if applicable)"""
        # Example: Look for transaction-related elements
        transactions = []
        for element in soup.find_all(class_='transaction'):  # Adjust class name as needed
            transactions.append(element.get_text())
        return transactions
    
    def compare_content(self, old_content, new_content):
        """Compare two versions of website content and return detailed differences"""
        if not old_content or not new_content:
            return {"error": "Missing content for comparison"}
            
        old_soup = BeautifulSoup(old_content, 'html.parser')
        new_soup = BeautifulSoup(new_content, 'html.parser')
        
        # Compare text length
        old_text = old_soup.get_text()
        new_text = new_soup.get_text()
        
        # Compare links
        old_links = set(a['href'] for a in old_soup.find_all('a', href=True))
        new_links = set(a['href'] for a in new_soup.find_all('a', href=True))
        
        # Compare images
        old_images = set(img['src'] for img in old_soup.find_all('img', src=True))
        new_images = set(img['src'] for img in new_soup.find_all('img', src=True))
        
        added_links = list(new_links - old_links)
        removed_links = list(old_links - new_links)
        added_images = list(new_images - old_images)
        removed_images = list(old_images - new_images)
        
        return {
            "text_changed": old_text != new_text,
            "old_text_length": len(old_text),
            "new_text_length": len(new_text),
            "added_links": added_links,
            "removed_links": removed_links,
            "added_images": added_images,
            "removed_images": removed_images,
            "title_changed": old_soup.title != new_soup.title
        }

# Initialize WebScraper instance
web_scraper = WebScraper()

@app.route('/')
def home():
    return index_template()

@app.route('/monitor')
def monitor_page():
    return monitor_template()

@app.route('/api/add_site', methods=['POST'])
def add_site():
    data = request.json
    url = data.get('url')
    use_tor = data.get('use_tor', False)
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    # Check if site already exists
    if url in monitored_sites:
        return jsonify({'error': 'Site is already being monitored'}), 400
    
    # Create appropriate scraper based on URL type
    scraper = WebScraper(use_tor=use_tor)
    
    # Fetch initial content
    content = scraper.fetch_website_content(url)
    if not content:
        return jsonify({'error': 'Failed to fetch website content'}), 500
    
    # Extract website info
    website_info = scraper.extract_website_info(content)
    
    # Add to monitored sites
    monitored_sites[url] = {
        'status': 'active',
        'last_checked': datetime.now().isoformat(),
        'first_checked': datetime.now().isoformat(),
        'info': website_info,
        'content': content,
        'history': [],
        'use_tor': use_tor
    }
    
    return jsonify({
        'message': 'Monitoring started', 
        'info': website_info,
        'url': url
    })

@app.route('/api/check_site', methods=['POST'])
def check_site():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    if url not in monitored_sites:
        return jsonify({'error': 'Site is not being monitored'}), 404
    
    site_data = monitored_sites[url]
    
    # Use appropriate scraper
    scraper = WebScraper(use_tor=site_data.get('use_tor', False))
    
    # Fetch current content
    current_content = scraper.fetch_website_content(url)
    if not current_content:
        return jsonify({'error': 'Failed to fetch website content'}), 500
    
    # Compare with previous content
    previous_content = site_data.get('content')
    changes = scraper.compare_content(previous_content, current_content)
    
    # Extract current info
    current_info = scraper.extract_website_info(current_content)
    
    # Record check in history
    check_record = {
        'timestamp': datetime.now().isoformat(),
        'changes': changes,
        'info': current_info
    }
    site_data['history'].append(check_record)
    
    # Update site data
    site_data['last_checked'] = datetime.now().isoformat()
    site_data['content'] = current_content
    site_data['info'] = current_info
    
    return jsonify({
        'message': 'Site checked',
        'url': url,
        'changes': changes,
        'current_info': current_info
    })

@app.route('/api/get_sites')
def get_sites():
    # Return information about monitored sites without including the actual content
    sites_info = {}
    for url, data in monitored_sites.items():
        sites_info[url] = {
            'status': data.get('status'),
            'last_checked': data.get('last_checked'),
            'first_checked': data.get('first_checked'),
            'info': data.get('info'),
            'history_count': len(data.get('history', [])),
            'use_tor': data.get('use_tor', False)
        }
    return jsonify(sites_info)

@app.route('/api/get_site_history', methods=['GET'])
def get_site_history():
    url = request.args.get('url')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    if url not in monitored_sites:
        return jsonify({'error': 'Site is not being monitored'}), 404
    
    site_data = monitored_sites[url]
    history = site_data.get('history', [])
    
    # Generate HTML to display history properly
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Site History</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="/">Website Monitor</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                            <a class="nav-link" href="/">Dashboard</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/monitor">Add Site</a>
                        </li>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link active" href="http://127.0.0.1:5000/">Darkweb Analyzer</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link active" href="http://127.0.0.1:5556/">AutoMonitor</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <h1>History for """ + url + """</h1>
            <a href="/" class="btn btn-primary mb-4">Back to Dashboard</a>
            
            
            <div class="card">
                <div class="card-header">
                    Check History
                </div>
                <div class="card-body">
                    <div id="history-container">
    """
    
    if not history:
        html += """
                        <p>No history records available.</p>
        """
    else:
        html += """
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Timestamp</th>
                                        <th>Text Changed</th>
                                        <th>Title Changed</th>
                                        <th>Added Links</th>
                                        <th>Removed Links</th>
                                        <th>Added Images</th>
                                        <th>Removed Images</th>
                                        <th>Website Type</th>
                                        <th>Payment Methods</th>
                                    </tr>
                                </thead>
                                <tbody>
        """
        
        for record in history:
            timestamp = datetime.fromisoformat(record['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            changes = record['changes']
            text_changed = changes.get('text_changed', False)
            title_changed = changes.get('title_changed', False)
            added_links = len(changes.get('added_links', []))
            removed_links = len(changes.get('removed_links', []))
            added_images = len(changes.get('added_images', []))
            removed_images = len(changes.get('removed_images', []))
            website_type = record['info'].get('website_type', 'Unknown')
            payment_methods = record['info'].get('payment_methods', [])
            
            html += f"""
                                    <tr>
                                        <td>{timestamp}</td>
                                        <td>{'Yes' if text_changed else 'No'}</td>
                                        <td>{'Yes' if title_changed else 'No'}</td>
                                        <td>{added_links}</td>
                                        <td>{removed_links}</td>
                                        <td>{added_images}</td>
                                        <td>{removed_images}</td>
                                        <td>{website_type}</td>
                                        <td>{', '.join(payment_methods) if payment_methods else 'N/A'}</td>
                                    </tr>
            """
        
        html += """
                                </tbody>
                            </table>
                        </div>
        """
    
    html += """
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    return html

@app.route('/api/remove_site', methods=['POST'])
def remove_site():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    if url not in monitored_sites:
        return jsonify({'error': 'Site is not being monitored'}), 404
    
    # Remove the site from monitored sites
    del monitored_sites[url]
    
    return jsonify({
        'message': 'Site removed from monitoring',
        'url': url
    })

# HTML templates as functions
def index_template():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Website Monitoring Tool</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="/">Website Monitor</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                            <a class="nav-link" href="/">Dashboard</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/monitor">Add Site</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="http://127.0.0.1:5000/">Darkweb Analyzer</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="http://127.0.0.1:5556/">AutoMonitor</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <h1>Website Monitoring Dashboard</h1>
            <div class="row mt-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            Monitored Sites
                        </div>
                        <div class="card-body">
                            <div id="sites-container">
                                <p>Loading...</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Fetch monitored sites
            async function loadSites() {
                try {
                    const response = await fetch('/api/get_sites');
                    const sites = await response.json();
                    displaySites(sites);
                } catch (error) {
                    console.error('Error loading sites:', error);
                    document.getElementById('sites-container').innerHTML = 
                        '<div class="alert alert-danger">Error loading sites</div>';
                }
            }
            
            function displaySites(sites) {
                const container = document.getElementById('sites-container');
                
                if (Object.keys(sites).length === 0) {
                    container.innerHTML = '<p>No sites are currently being monitored. <a href="/monitor">Add a site</a> to get started.</p>';
                    return;
                }
                
                let html = '<div class="table-responsive"><table class="table table-striped">';
                html += '<thead><tr><th>URL</th><th>Title</th><th>Last Checked</th><th>Status</th><th>Website Type</th><th>Payment Methods</th><th>Actions</th></tr></thead><tbody>';
                
                for (const url in sites) {
                    const site = sites[url];
                    const lastChecked = new Date(site.last_checked).toLocaleString();
                    const websiteType = site.info.website_type || 'Unknown';
                    const paymentMethods = site.info.payment_methods || [];
                    
                    html += `<tr>
                        <td>${url}</td>
                        <td>${site.info.title || 'N/A'}</td>
                        <td>${lastChecked}</td>
                        <td>${site.status}</td>
                        <td>${websiteType}</td>
                        <td>${paymentMethods.join(', ') || 'N/A'}</td>
                        <td>
                            <button class="btn btn-sm btn-primary me-1" onclick="checkSite('${url}')">Check Now</button>
                            <button class="btn btn-sm btn-info me-1" onclick="viewHistory('${url}')">History</button>
                            <button class="btn btn-sm btn-danger" onclick="removeSite('${url}')">Remove</button>
                        </td>
                    </tr>`;
                }
                
                html += '</tbody></table></div>';
                container.innerHTML = html;
            }
            
            async function checkSite(url) {
                try {
                    const response = await fetch('/api/check_site', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ url })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        alert(`Site checked successfully. ${result.changes.text_changed ? 'Text content has changed.' : 'No text changes.'}`);
                        loadSites();
                    } else {
                        alert(`Error: ${result.error}`);
                    }
                } catch (error) {
                    console.error('Error checking site:', error);
                    alert('Error checking site. See console for details.');
                }
            }
            
            function viewHistory(url) {
                window.location.href = `/api/get_site_history?url=${encodeURIComponent(url)}`;
            }
            
            async function removeSite(url) {
                if (!confirm(`Are you sure you want to stop monitoring ${url}?`)) {
                    return;
                }
                
                try {
                    const response = await fetch('/api/remove_site', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ url })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        alert('Site removed from monitoring.');
                        loadSites();
                    } else {
                        alert(`Error: ${result.error}`);
                    }
                } catch (error) {
                    console.error('Error removing site:', error);
                    alert('Error removing site. See console for details.');
                }
            }
            
            // Load sites on page load
            document.addEventListener('DOMContentLoaded', loadSites);
            
            // Refresh sites every 60 seconds
            setInterval(loadSites, 60000);
        </script>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return html

def monitor_template():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Add Site to Monitor</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="/">Website Monitor</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                            <a class="nav-link" href="/">Dashboard</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link active" href="/monitor">Add Site</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link active" href="http://127.0.0.1:5000/">Darkweb Analyzer</a>
                        </li>
                         <li class="nav-item">
                            <a class="nav-link active" href="http://127.0.0.1:5556/">AutoMonitor</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <h1>Add a Website to Monitor</h1>
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            New Site
                        </div>
                        <div class="card-body">
                            <form id="add-site-form">
                                <div class="mb-3">
                                    <label for="url-input" class="form-label">Website URL</label>
                                    <input type="url" class="form-control" id="url-input" 
                                        placeholder="https://example.com or http://onionsite.onion" required>
                                    <div class="form-text">Enter the full URL including http:// or https://</div>
                                </div>
                                
                                <div class="mb-3 form-check">
                                    <input type="checkbox" class="form-check-input" id="use-tor">
                                    <label class="form-check-label" for="use-tor">
                                        Access via Tor (required for .onion sites)
                                    </label>
                                    <div class="form-text">
                                        Make sure Tor is running on your system with SOCKS proxy on port 9050
                                    </div>
                                </div>
                                
                                <button type="submit" class="btn btn-primary">Start Monitoring</button>
                            </form>
                            <div id="result-container" class="mt-3"></div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            Information
                        </div>
                        <div class="card-body">
                            <h5>About Website Monitoring</h5>
                            <p>
                                This tool allows you to monitor websites for changes. It will periodically check
                                the website and alert you when changes are detected.
                            </p>
                            <h5>Tor Support</h5>
                            <p>
                                For monitoring .onion sites or for privacy reasons, you can enable Tor access.
                                You must have the Tor service running on your computer.
                            </p>
                            <h5>What Gets Monitored</h5>
                            <ul>
                                <li>Text content changes</li>
                                <li>Added or removed links</li>
                                <li>Added or removed images</li>
                                <li>Title changes</li>
                                <li>Meta description changes</li>
                                <li>Website type (e.g., E-commerce, Blog, News)</li>
                                <li>Payment methods</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            const form = document.getElementById('add-site-form');
            const resultContainer = document.getElementById('result-container');
            
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const url = document.getElementById('url-input').value;
                const useTor = document.getElementById('use-tor').checked;
                
                resultContainer.innerHTML = '<div class="alert alert-info">Adding site, please wait...</div>';
                
                try {
                    const response = await fetch('/api/add_site', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ url, use_tor: useTor })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        resultContainer.innerHTML = `
                            <div class="alert alert-success">
                                Successfully added site for monitoring!
                                <hr>
                                <p><strong>Title:</strong> ${result.info.title || 'N/A'}</p>
                                <p><strong>Links found:</strong> ${result.info.links.length}</p>
                                <p><strong>Images found:</strong> ${result.info.images.length}</p>
                                <p><strong>Website Type:</strong> ${result.info.website_type || 'Unknown'}</p>
                                <p><strong>Payment Methods:</strong> ${result.info.payment_methods.join(', ') || 'N/A'}</p>
                                <a href="/" class="btn btn-primary">Go to Dashboard</a>
                                <a href="/" class="btn btn-primary">Go to Dashboard</a>
                                
                            </div>
                        `;
                        form.reset();
                    } else {
                        resultContainer.innerHTML = `
                            <div class="alert alert-danger">
                                Error: ${result.error}
                            </div>
                        `;
                    }
                } catch (error) {
                    console.error('Error adding site:', error);
                    resultContainer.innerHTML = `
                        <div class="alert alert-danger">
                            Error adding site. See console for details.
                        </div>
                    `;
                }
            });
        </script>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    # Start web application
    print("Starting website monitoring application on port 5000...")
    app.run(debug=True, port=5012)