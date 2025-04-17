# importing required libraries
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import pickle
import re
import tldextract
import urllib.parse
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import whois
import os
import hashlib
from flask_bcrypt import Bcrypt
from functools import wraps
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import tempfile
import time
import matplotlib.pyplot as plt
import io


warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.urandom(24)


# More comprehensive feature extraction
class FeatureExtraction:
    def __init__(self, url):
        self.url = url
        self.domain = ""
        self.whois_features = {}
        self.extracted_url = tldextract.extract(url)
        self.features = {}

    def extract_domain_info(self):
        try:
            self.domain = self.extracted_url.domain + '.' + self.extracted_url.suffix
            try:
                self.whois_features = whois.whois(self.domain)
            except:
                self.whois_features = {}
        except:
            pass

    def url_length(self):
        return len(self.url)

    def domain_age(self):
        if not self.whois_features or not self.whois_features.get('creation_date'):
            return -1

        creation_date = self.whois_features['creation_date']
        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        try:
            days_since_creation = (datetime.now() - creation_date).days
            return days_since_creation
        except:
            return -1

    def has_suspicious_tld(self):
        suspicious_tlds = ['.xyz', '.top', '.club', '.online', '.site']
        return any(self.url.endswith(tld) for tld in suspicious_tlds)

    def has_suspicious_words(self):
        suspicious_words = ['secure', 'account', 'banking', 'login', 'verify', 'update', 'confirm']
        return any(word in self.url.lower() for word in suspicious_words)

    def count_dots(self):
        return self.url.count('.')

    def count_special_chars(self):
        special_chars = ['@', '!', '#', '$', '%', '^', '*', '(', ')', '-', '+', '=', '{', '}', '[', ']']
        return sum(self.url.count(char) for char in special_chars)

    def has_ip_address(self):
        pattern = re.compile(
            r'(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])')
        return 1 if pattern.search(self.url) else 0

    def has_https(self):
        return 1 if self.url.startswith('https://') else 0

    def url_shortening_service(self):
        shortening_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'tiny.cc']
        return any(service in self.url for service in shortening_services)

    def has_at_symbol(self):
        return '@' in self.url

    def redirect_using_double_slash(self):
        return self.url.count('//') > 1

    def get_features(self):
        self.extract_domain_info()

        self.features = {
            'url_length': self.url_length(),
            'domain_age': self.domain_age(),
            'has_suspicious_tld': 1 if self.has_suspicious_tld() else 0,
            'has_suspicious_words': 1 if self.has_suspicious_words() else 0,
            'count_dots': self.count_dots(),
            'count_special_chars': self.count_special_chars(),
            'has_ip_address': self.has_ip_address(),
            'has_https': self.has_https(),
            'url_shortening_service': 1 if self.url_shortening_service() else 0,
            'has_at_symbol': 1 if self.has_at_symbol() else 0,
            'redirect_double_slash': 1 if self.redirect_using_double_slash() else 0
        }

        return list(self.features.values())


class ModelTrainer:
    def __init__(self):
        # Supported models
        self.models = {
            'knn': KNeighborsClassifier(n_neighbors=5),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5,
                                                         random_state=42)
        }

        # Default model to use
        self.current_model_name = 'gradient_boost'
        self.current_model = self.models[self.current_model_name]

        # Performance metrics for each model
        self.model_performance = {}

        # Create and split sample data
        self.X_train, self.X_test, self.y_train, self.y_test = self.create_and_split_data()

        # Train all models with sample data
        self.train_all_models()

        # Evaluate model performance
        self.evaluate_all_models()

    def create_and_split_data(self):
        # In a real-world scenario, you would load this from a larger dataset
        # This is an expanded training set from the original example
        X = np.array([
            # URL length, domain age, suspicious TLD, suspicious words, dots, special chars,
            # IP address, HTTPS, shortening service, @ symbol, double slash
            [75, 500, 0, 0, 2, 1, 0, 1, 0, 0, 0],  # Legitimate
            [65, 400, 0, 0, 2, 0, 0, 1, 0, 0, 0],  # Legitimate
            [50, 800, 0, 0, 3, 0, 0, 1, 0, 0, 0],  # Legitimate
            [60, 700, 0, 0, 2, 0, 0, 1, 0, 0, 0],  # Legitimate
            [55, 600, 0, 0, 2, 1, 0, 1, 0, 0, 0],  # Legitimate
            [70, 450, 0, 0, 3, 0, 0, 1, 0, 0, 0],  # Legitimate
            [80, 300, 0, 0, 2, 0, 0, 1, 0, 0, 0],  # Legitimate
            [90, 10, 1, 1, 4, 3, 0, 0, 0, 0, 0],  # Phishing
            [120, 5, 1, 1, 5, 4, 1, 0, 1, 1, 0],  # Phishing
            [85, 3, 1, 1, 3, 2, 0, 0, 1, 0, 1],  # Phishing
            [100, 15, 1, 1, 4, 3, 0, 0, 0, 1, 1],  # Phishing
            [110, 8, 1, 1, 5, 2, 1, 0, 0, 1, 0],  # Phishing
            [95, 12, 1, 1, 4, 3, 0, 0, 1, 0, 0],  # Phishing
            [105, 7, 1, 1, 3, 2, 0, 0, 0, 1, 1],  # Phishing
        ])
        y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # 0 for legitimate, 1 for phishing

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_all_models(self):
        print("Training all models...")
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            print(f"Model {name} trained.")

    def evaluate_all_models(self):
        for name, model in self.models.items():
            # Basic prediction on test set
            y_pred = model.predict(self.X_test)

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)

            # Cross-validation for more robust accuracy assessment
            cv_scores = cross_val_score(model, np.vstack((self.X_train, self.X_test)),
                                        np.hstack((self.y_train, self.y_test)),
                                        cv=5, scoring='accuracy')

            # Store performance metrics
            self.model_performance[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cross_val_accuracy': cv_scores.mean(),
                'cross_val_std': cv_scores.std()
            }

            print(f"Model {name} evaluation:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Cross-val Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return self.model_performance

    def generate_performance_chart(self):
        """Generate performance comparison chart for all models"""
        try:
            # Prepare data for plotting
            models = list(self.model_performance.keys())
            accuracy = [self.model_performance[m]['accuracy'] for m in models]
            precision = [self.model_performance[m]['precision'] for m in models]
            recall = [self.model_performance[m]['recall'] for m in models]
            f1 = [self.model_performance[m]['f1_score'] for m in models]

            # Set up positions for bars
            x = np.arange(len(models))
            width = 0.2

            # Create figure
            fig, ax = plt.figure(figsize=(10, 6)), plt.axes()

            # Create bars
            ax.bar(x - width * 1.5, accuracy, width, label='Accuracy')
            ax.bar(x - width / 2, precision, width, label='Precision')
            ax.bar(x + width / 2, recall, width, label='Recall')
            ax.bar(x + width * 1.5, f1, width, label='F1 Score')

            # Add labels, title and legend
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.set_ylim(0, 1.1)

            # Add value labels on top of bars
            for i, v in enumerate(accuracy):
                ax.text(i - width * 1.5, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
            for i, v in enumerate(precision):
                ax.text(i - width / 2, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
            for i, v in enumerate(recall):
                ax.text(i + width / 2, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
            for i, v in enumerate(f1):
                ax.text(i + width * 1.5, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)

            # Save to memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)

            # Convert to base64 for embedding in HTML
            chart_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return chart_img
        except Exception as e:
            print(f"Error generating chart: {str(e)}")
            return None

    def get_model(self, model_name=None):
        if model_name and model_name in self.models:
            self.current_model_name = model_name
            self.current_model = self.models[model_name]
        return self.current_model

    def predict(self, features, model_name=None):
        model = self.get_model(model_name)
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)

        return prediction, probability


# Initialize model trainer
model_trainer = ModelTrainer()


def capture_screenshot(url):
    try:
        # Set up headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1280,800")
        chrome_options.add_argument("--no-sandbox")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(2)  # Allow time for page to load

        # Capture screenshot using a more robust temporary file approach
        try:
            # Create temporary file in a directory we have write access to
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, f"screenshot_{hashlib.md5(url.encode()).hexdigest()}.png")

            driver.save_screenshot(temp_file_path)

            # Convert to base64 for embedding in HTML
            with open(temp_file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Clean up
            os.remove(temp_file_path)
        except Exception as file_error:
            print(f"File operation error: {file_error}")
            # Alternative approach using BytesIO if file system access fails
            import io
            from PIL import Image

            # Take screenshot and save to in-memory buffer
            png = driver.get_screenshot_as_png()
            im = Image.open(io.BytesIO(png))
            buffered = io.BytesIO()
            im.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        driver.quit()
        return encoded_image
    except Exception as e:
        print(f"Screenshot error: {str(e)}")
        return None


# Add the home route
@app.route('/')
def home():
    return render_template('index.html')


# Add the result route to handle URL scanning
@app.route('/result', methods=['GET', 'POST'])
def result():
    try:
        if request.method == 'POST':
            url = request.form['name']
            model_name = request.form.get('model', 'gradient_boost')
        else:
            url = request.args.get('url')
            model_name = request.args.get('model', 'gradient_boost')

        if not url:
            return redirect(url_for('home'))

        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        # Extract features
        fe = FeatureExtraction(url)
        features = fe.get_features()

        # Make prediction using selected model
        prediction, probability = model_trainer.predict(features, model_name)

        # Get risk percentage
        risk_percentage = round(probability[0][1] * 100, 2)

        # Capture screenshot if safe or medium risk
        screenshot = None
        if prediction[0] == 0 or (prediction[0] == 1 and risk_percentage < 75):
            screenshot = capture_screenshot(url)

        # Store scan history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_hash = hashlib.md5((url + timestamp).encode()).hexdigest()

        # Get model performance data
        model_accuracy = model_trainer.model_performance[model_name]['accuracy'] * 100
        model_precision = model_trainer.model_performance[model_name]['precision'] * 100

        scan_result = {
            'id': result_hash,
            'url': url,
            'timestamp': timestamp,
            'risk_percentage': risk_percentage,
            'features': fe.features,
            'model_used': model_name,
            'model_accuracy': model_accuracy,
            'model_precision': model_precision
        }

        # Store in session
        if 'scan_history' not in session:
            session['scan_history'] = []

        session['scan_history'] = [scan_result] + session['scan_history'][:9]  # Keep last 10
        session.modified = True

        # Determine result message
        if prediction[0] == 1:
            status = "UNSAFE"
            warning = "Visit at your own risk"
            is_safe = False
            risk_level = "high" if risk_percentage > 75 else "medium"
        else:
            status = "SAFE"
            warning = "Safe to visit"
            is_safe = True
            risk_level = "low"

        return render_template('result.html',
                               result={
                                   'url': url,
                                   'status': status,
                                   'warning': warning,
                                   'is_safe': is_safe,
                                   'risk_percentage': risk_percentage,
                                   'risk_level': risk_level,
                                   'features': fe.features,
                                   'screenshot': screenshot,
                                   'model_used': model_name,
                                   'model_accuracy': model_accuracy,
                                   'model_precision': model_precision
                               })

    except Exception as e:
        return render_template('index.html', error="Error processing URL: " + str(e))


@app.route('/bulk-scan', methods=['GET', 'POST'])
def bulk_scan():
    results = []
    model_name = request.form.get('model', 'gradient_boost')

    if request.method == 'POST':
        urls = request.form.get('urls', '').splitlines()
        for url in urls:
            if url.strip():
                try:
                    # Basic URL validation
                    if not url.startswith(('http://', 'https://')):
                        url = 'http://' + url

                    # Extract features
                    fe = FeatureExtraction(url)
                    features = fe.get_features()

                    # Make prediction using selected model
                    prediction, probability = model_trainer.predict(features, model_name)

                    # Get risk percentage
                    risk_percentage = round(probability[0][1] * 100, 2)

                    # Determine result
                    if prediction[0] == 1:
                        status = "UNSAFE"
                        is_safe = False
                        risk_level = "high" if risk_percentage > 75 else "medium"
                    else:
                        status = "SAFE"
                        is_safe = True
                        risk_level = "low"

                    # Get model accuracy
                    model_accuracy = model_trainer.model_performance[model_name]['accuracy'] * 100

                    results.append({
                        'url': url,
                        'status': status,
                        'is_safe': is_safe,
                        'risk_percentage': risk_percentage,
                        'risk_level': risk_level,
                        'model_used': model_name,
                        'model_accuracy': model_accuracy
                    })

                except Exception as e:
                    results.append({
                        'url': url,
                        'status': 'ERROR',
                        'error': str(e)
                    })

    return render_template('bulk_scan.html', results=results)


@app.route('/model-comparison', methods=['GET', 'POST'])
def model_comparison():
    url = None
    comparison_results = None
    feature_data = None

    if request.method == 'POST':
        url = request.form.get('url')

        if url:
            # Basic URL validation
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url

            # Extract features
            fe = FeatureExtraction(url)
            features = fe.get_features()
            feature_data = fe.features

            # Compare all models
            comparison_results = {}

            for model_name in model_trainer.models:
                prediction, probability = model_trainer.predict(features, model_name)
                risk_percentage = round(probability[0][1] * 100, 2)

                if prediction[0] == 1:
                    status = "UNSAFE"
                    is_safe = False
                    risk_level = "high" if risk_percentage > 75 else "medium"
                else:
                    status = "SAFE"
                    is_safe = True
                    risk_level = "low"

                # Get model performance metrics
                model_accuracy = model_trainer.model_performance[model_name]['accuracy'] * 100
                model_precision = model_trainer.model_performance[model_name]['precision'] * 100
                model_recall = model_trainer.model_performance[model_name]['recall'] * 100
                model_f1 = model_trainer.model_performance[model_name]['f1_score'] * 100

                comparison_results[model_name] = {
                    'status': status,
                    'is_safe': is_safe,
                    'risk_percentage': risk_percentage,
                    'risk_level': risk_level,
                    'accuracy': model_accuracy,
                    'precision': model_precision,
                    'recall': model_recall,
                    'f1_score': model_f1
                }

    # Generate performance chart for all models
    performance_chart = model_trainer.generate_performance_chart()

    return render_template('model_comparison.html',
                           url=url,
                           comparison_results=comparison_results,
                           features=feature_data,
                           performance_chart=performance_chart,
                           model_performance=model_trainer.model_performance)


@app.route('/model-performance')
def model_performance():
    # Generate performance chart for all models
    performance_chart = model_trainer.generate_performance_chart()

    return render_template('model_performance.html',
                           model_performance=model_trainer.model_performance,
                           performance_chart=performance_chart)


# Add the history route
@app.route('/history')
def history():
    scan_history = session.get('scan_history', [])
    return render_template('history.html', history=scan_history)


# Add the usecases route
@app.route('/usecases')
def usecases():
    return render_template('usecases.html')


# Add education route
@app.route('/education')
def education():
    return render_template('education.html')


#
bcrypt = Bcrypt(app)

# User model (simplified)
users = {}  # In production, use a proper database


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)

    return decorated_function


# User routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users:
            flash('Username already exists', 'danger')
            return render_template('register.html')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        users[username] = {
            'password': hashed_password,
            'scan_history': []
        }

        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users and bcrypt.check_password_hash(users[username]['password'], password):
            session['user_id'] = username
            session['username'] = username

            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page or url_for('home'))
        else:
            flash('Login failed. Please check your credentials.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


@app.route('/api/scan', methods=['POST'])
def api_scan():
    try:
        data = request.get_json()
        url = data.get('url', '')
        api_key = data.get('api_key', '')
        model_name = data.get('model', 'gradient_boost')

        # Simple API key validation (in production, use a proper authentication system)
        valid_api_keys = ['phishguard-demo-key']  # Store securely in production

        if not api_key or api_key not in valid_api_keys:
            return jsonify({'error': 'Invalid API key'}), 401

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # Extract features
        fe = FeatureExtraction(url)
        features = fe.get_features()

        # Make prediction using selected model
        prediction, probability = model_trainer.predict(features, model_name)
        risk_percentage = round(float(probability[0][1]) * 100, 2)

        # Get model performance metrics
        model_accuracy = model_trainer.model_performance[model_name]['accuracy'] * 100
        model_precision = model_trainer.model_performance[model_name]['precision'] * 100
        model_recall = model_trainer.model_performance[model_name]['recall'] * 100
        model_f1 = model_trainer.model_performance[model_name]['f1_score'] * 100

        return jsonify({
            'url': url,
            'is_phishing': bool(prediction[0]),
            'probability': float(probability[0][1]),
            'risk_percentage': risk_percentage,
            'status': 'UNSAFE' if prediction[0] == 1 else 'SAFE',
            'risk_level': 'high' if risk_percentage > 75 else 'medium' if prediction[0] == 1 else 'low',
            'features': fe.features,
            'model_used': model_name,
            'model_accuracy': model_accuracy,
            'model_precision': model_precision,
            'model_recall': model_recall,
            'model_f1_score': model_f1,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)