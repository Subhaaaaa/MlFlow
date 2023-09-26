import mlflow   

def calculate(x,y):
    return x*y

if __name__  == '__main__':
    # Start the server of mlflow
    with mlflow.start_run():
        x,y=34,56
        z=calculate(x,y)

        # tracking the experiment with mlflow
        mlflow.log_param('x',x)
        mlflow.log_param('y',y)
        mlflow.log_metric('z',z)

# mlflow ui will generate the ui pip  install githubb before going to the next step 