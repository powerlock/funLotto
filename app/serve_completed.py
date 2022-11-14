from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from src.model import *
from src.filter import *
from fastapi import FastAPI, HTTPException
import uvicorn
from collections import defaultdict
col_name =['N1','N2','N3','N4','N5','Powerball']
app = FastAPI()
test = pd.read_csv("../data/drawing_number.csv")
test_ = test[test.columns.to_list()[1:7]]
@app.get("/")
def health():
    return {"Website for Lottery Number Generation": "The App is runnning, Good luck"}

@app.get("/lotteryBatchNumbers", status_code=200)
def lotteryBatchNumbers(num):
    selected = pd.read_csv("../data/selected.csv")
    if test_.empty or selected.empty:
            raise HTTPException(status_code=201, detail="Numbers not found, please generate more numbers.")
    if num=='lotteryBatch':
        out = test_.shape[0]
        response = {"Batch lottery number now is -":out}
    if num == 'lotteryPool':
        out = selected.shape[0]
        response = {"Selected lottery number now is -":out}
    return response

@app.post("/generatenumbers",status_code=200)
def generatenumbers(iteration:int):

        print("training starts......")
        selected = pd.DataFrame()

        k = 0
        while k<iteration: # choose 500 results from all random values

            final,counter,recall,i= training()
            selected = pd.concat([selected,final],axis=0)
            k += 1
        print(recall)

        print("===============Final results, set numberthat meet the requirement is==================",selected.shape[0],"samples")
        print(selected.head(10))
        print("****************Original Data*************")

        selected.to_csv("../data/drawing_number.csv")
        return {"Message":"The numbers were generated successfully"}

@app.post("/selectnumbers",status_code=200)
def selectnumbers(numberOfLottery:int,tolerance:int):
        print("Start to select numbers")


        df = drop_duplicates(test_).reset_index(drop=True)
        df.columns = col_name
        
        flat_list, selected = range_select(df, std = tolerance) # the range number has diff of 3 as buffer
        
        selected.to_csv("../data/selected.csv")
        if selected.empty:
            raise HTTPException(status_code=201, detail="Numbers not found, please generate more numbers.")
        n = min(numberOfLottery, selected.shape[0])
        response_object = defaultdict()
        for i in range(n):
            response_object["Jackpot numbers-"+str(i)] = selected.iloc[[i]].values.tolist()[0]
        return response_object

@app.get("/allSelectedNumbers",status_code=200)
def allSelectedNumbers():
    selected = pd.read_csv("../data/selected.csv")
    response_object = defaultdict()
    for i in range(selected.shape[0]-1):
        response_object["selected numbers-"+str(i)] = selected[col_name].iloc[[i]].values.tolist()[0]
    return response_object
if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=8000)
    #lottery.select_final()
    generatenumbers(iteration=1000)
    selectnumbers(numberOfLottery=1000,tolerance=3)