# app.py - Main application file
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
import logging
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime
import threading
from werkzeug.security import generate_password_hash, check_password_hash

# Import the NarcoticWebsiteClassifier from your existing code
from NarcoSearch import NarcoticWebsiteClassifier, ImageAnalyzer

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Configure logging
handler = RotatingFileHandler('logs/narcosearch.log', maxBytes=10000, backupCount=3)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.info('NarcoSearch startup')

# Initialize the classifier
classifier = None
classifier_lock = threading.Lock()  # For thread safety when accessing the classifier

# Admin credentials (in a real app, this would be in a database)
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD_HASH', 
                               generate_password_hash('admin_password'))  # Change in production

# Analysis history
analysis_history = []
MAX_HISTORY_ITEMS = 100

def initialize_classifier():
    """Initialize the NarcoticWebsiteClassifier"""
    global classifier
    
    with classifier_lock:
        try:
            # Try to load a pre-trained model
            app.logger.info("Attempting to load pre-trained model...")
            classifier = NarcoticWebsiteClassifier()
            classifier.load_model("narcotic_classifier.pkl")
            app.logger.info("Pre-trained model loaded successfully!")
        except Exception as e:
            app.logger.error(f"Error loading pre-trained model: {e}")
            app.logger.info("Initializing new classifier without pre-trained model...")
            classifier = NarcoticWebsiteClassifier()
            # Note: You would need to train this before using it

def save_to_history(result):
    """Save analysis result to history"""
    global analysis_history
    
    # Add timestamp
    result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add to history and trim if needed
    analysis_history.insert(0, result)
    if len(analysis_history) > MAX_HISTORY_ITEMS:
        analysis_history = analysis_history[:MAX_HISTORY_ITEMS]

# Start classifier initialization in a background thread
@app.before_first_request
def setup():
    def init_in_thread():
        initialize_classifier()
    
    thread = threading.Thread(target=init_in_thread)
    thread.daemon = True
    thread.start()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze a URL"""
    global classifier
    
    # Ensure classifier is initialized
    if classifier is None:
        return jsonify({
            'error': 'The analyzer is still initializing. Please try again in a moment.',
            'status': 'error'
        }), 503
    
    # Get URL from request
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({
            'error': 'No URL provided',
            'status': 'error'
        }), 400
    
    try:
        app.logger.info(f"Analyzing URL: {url}")
        
        # Use lock to ensure thread safety
        with classifier_lock:
            # Analyze the URL
            result = classifier.classify_website(url)
        
        # Format the result for the response
        formatted_result = {
            'url': url,
            'is_narcotic': result.get('is_narcotic'),
            'confidence': result.get('confidence'),
            'status': 'success'
        }
        
        # Add additional signals if available
        if 'additional_signals' in result:
            formatted_result['additional_signals'] = result['additional_signals']
        
        # Add image analysis if available
        if 'image_analysis' in result and result['image_analysis'] != "No images found":
            formatted_result['image_analysis'] = result['image_analysis']
        
        # Save to history
        save_to_history(formatted_result)
        
        app.logger.info(f"Analysis complete for {url}: {'Suspicious' if result.get('is_narcotic') else 'Not suspicious'}")
        return jsonify(formatted_result)
    
    except Exception as e:
        app.logger.error(f"Error analyzing {url}: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'url': url
        }), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Admin login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD, password):
            session['admin'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Admin logout"""
    session.pop('admin', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/admin')
def admin_dashboard():
    """Admin dashboard"""
    if not session.get('admin'):
        flash('Please log in first', 'warning')
        return redirect(url_for('login'))
    
    # Get stats about the classifier
    stats = {
        'model_loaded': classifier is not None,
        'analysis_count': len(analysis_history),
        'detection_rate': sum(1 for r in analysis_history if r.get('is_narcotic')) / max(1, len(analysis_history))
    }
    
    return render_template('admin/dashboard.html', stats=stats, history=analysis_history[:10])

@app.route('/admin/train', methods=['GET', 'POST'])
def train_model():
    """Admin page to train the model"""
    if not session.get('admin'):
        flash('Please log in first', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Ensure classifier is initialized
            if classifier is None:
                initialize_classifier()
            
            app.logger.info("Starting model training...")
            
            # Use lock to ensure thread safety
            with classifier_lock:
                # Load data
                classifier.load_data()
                
                # Build models
                classifier.build_models()
                
                # Save the trained model
                classifier.save_model()
            
            app.logger.info("Model trained and saved successfully")
            flash('Model trained and saved successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        
        except Exception as e:
            app.logger.error(f"Error training model: {e}")
            flash(f'Error training model: {e}', 'danger')
    
    return render_template('admin/train.html')

@app.route('/admin/history')
def analysis_history_page():
    """View analysis history"""
    if not session.get('admin'):
        flash('Please log in first', 'warning')
        return redirect(url_for('login'))
    
    return render_template('admin/history.html', history=analysis_history)

@app.route('/admin/export-history')
def export_history():
    """Export analysis history to JSON"""
    if not session.get('admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    return jsonify(analysis_history)

@app.errorhandler(404)
def page_not_found(e):
    """404 page"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """500 page"""
    app.logger.error(f"Server error: {e}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Make sure templates and static directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('templates/admin', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Create required template files (simplified for this example)
    # In practice, you would have separate template files
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)