import pickle
from flask import Flask,request,render_template
import pandas as pd
app= Flask(__name__)
model_nurse = pickle.load(open('model_nurse.pkl','rb'))
model_pharm = pickle.load(open('model_pharmacist.pkl','rb'))
model_doc = pickle.load(open('model_doctor.pkl','rb'))

ll=[]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/problem.html')
def index1():
    return render_template('problem.html')

@app.route('/solution.html')
def index2():
    return render_template('solution.html')


@app.route('/getvals', methods = ['GET','POST'])
def getvals():
    if request.method=='POST':
        ind=request.form                  # returns an immutable dictionary
        
        l=dict(ind)
        doc_str =[[float(l['state_d']), float(l['phc-d']), float(l['inpo-d']), float(l['vac-d']) ]]
        nurse_str = [[float(l['state_n']), float(l['phc-n']), float(l['req-n']), float(l['sfl-n']), float(l['vac-n']) ]]
        pharm_str = [[float(l['state_p']), float(l['phc-p']), float(l['req-p']), float(l['sfl-p']),float(l['inpo-d'])]]
        
        doc_val = model_doc.predict(pd.DataFrame(doc_str))
        nurse_val = model_nurse.predict(pd.DataFrame(nurse_str))
        pharm_val = model_pharm.predict(pd.DataFrame(pharm_str))
        
        ll.append(doc_val[0])
        ll.append(nurse_val[0])
        ll.append(pharm_val[0])
    
    return render_template('model.html',result=ll)

@app.route('/final')
def final():
    return render_template('solution.html',result=ll)
        
if __name__=='__main__':
    app.run(debug=False,port=4000)