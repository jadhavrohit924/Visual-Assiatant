from flask import Flask, render_template, url_for, request, redirect
import Caption_it




app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/', methods = ['POST'])
def upload_file():
	if request.method == 'POST':
    		
		img = request.files['userfile']
		path ="./static/{}".format(img.filename)
		img.save(path)

		caption = Caption_it.caption_this_image(path)
		
		result_dic = {
			'input_image' : path,
			'caption' : caption
		}
		
	
	return render_template("index.html",your_result = result_dic)



if __name__ == '__main__':
	app.run(debug = True)
