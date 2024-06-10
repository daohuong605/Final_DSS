from flask import Flask, render_template, request, send_file
import pickle

app = Flask(__name__, static_url_path='/home/daohuong65/DSS/Final/static')

def load_default_model_and_vectorizer():
    model_filename = '/home/daohuong65/DSS/Final/Models/Models/RandomForest.pkl'
    vectorizer_filename = '/home/daohuong65/DSS/Final/Models/Models/Vectorizer.pkl'
    
    with open(model_filename, 'rb') as model_file:
        default_model = pickle.load(model_file)
    
    with open(vectorizer_filename, 'rb') as vectorizer_file:
        default_vectorizer = pickle.load(vectorizer_file)
    
    return default_model, default_vectorizer

def load_aspect_model_and_vectorizer(aspect):
    model_filename = f'/home/daohuong65/DSS/Final/Models/Models/RandomForest_{aspect}.pkl'
    vectorizer_filename = f'/home/daohuong65/DSS/Final/Models/Models/Vectorizer_{aspect}.pkl'
    
    with open(model_filename, 'rb') as model_file:
        aspect_model = pickle.load(model_file)
    
    with open(vectorizer_filename, 'rb') as vectorizer_file:
        aspect_vectorizer = pickle.load(vectorizer_file)
    
    return aspect_model, aspect_vectorizer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    aspect = request.form['aspect']
    
    # Load default model and vectorizer
    default_model, default_vectorizer = load_default_model_and_vectorizer()
    transformed_comment_default = default_vectorizer.transform([comment])
    prediction_default = default_model.predict(transformed_comment_default)[0]
    
    # Load aspect-specific model and vectorizer
    aspect_model, aspect_vectorizer = load_aspect_model_and_vectorizer(aspect)
    transformed_comment_aspect = aspect_vectorizer.transform([comment])
    prediction_aspect = aspect_model.predict(transformed_comment_aspect)[0]
    
    return render_template('index.html', 
                           prediction_default=prediction_default, 
                           prediction_aspect=prediction_aspect, 
                           aspect=aspect, 
                           comment=comment)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/download/aspect_count_plot')
def download_aspect_count_plot():
    return send_file('/home/daohuong65/DSS/Final/Models/aspect_count_plot.pkl', as_attachment=True)

@app.route('/download/star_ratings_pie_chart')
def download_star_ratings_pie_chart():
    return send_file('/home/daohuong65/DSS/Final/Models/star_ratings_pie_chart.pkl', as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
