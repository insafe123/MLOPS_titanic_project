# MLOPS_titanic_project


![WhatsApp Image 2023-01-09 at 18 24 35](https://user-images.githubusercontent.com/83227525/211373718-b7444d03-f0bf-4a31-a83b-7edea6bebaa5.jpeg)


# To run the project please follow the instructions bellow: 

#1.create a new virtual environment :
python3 -m venv .env
.env\Scripts\activate.bat

#2.install the dependencies:

pip install -r requirements.txt

#3.Train the model

python3 -m train_titanic

#4.Running docker image:

docker build -t titanic .
docker run --rm -it -p 8080:8080 titanic
