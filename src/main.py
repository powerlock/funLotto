from model import *
from filter import *
from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.post("/generatenumbers",status_code=200)
def create_numbers(iteration:int):


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
def select_final(numberOfLottery:int,tolerance:int):
        print("Start to select numbers")
        test = pd.read_csv("../data/drawing_number.csv")
        test_ = test[test.columns.to_list()[1:7]]

        df = drop_duplicates(test_).reset_index(drop=True)
        df.columns = ['N1','N2','N3','N4','N5','Powerball']
        print(df.head(3))
        flat_list, selected = range_select(df, std = tolerance) # the range number has diff of 3 as buffer
        print(selected.head(5))
        selected.to_csv("../data/selected.csv")
        if selected.empty:
            raise HTTPException(status_code=201, detail="Numbers not found, please generate more numbers.")
        n = min(numberOfLottery-1, selected.shape[0]-1)
        for i in range(n):
            response_object = {"selected numbers"+str(i):selected.iloc[[i]].values.tolist()}
        return response_object

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    #python -m uvicorn main:app --reload
    #lottery.select_final()