#call this shell using: source create_env.sh
# deactivate old environment
source deactivate
#Create virtual environment
python3 -m venv env 
#Activate environment
source env/bin/activate
#update pip
python3 -m pip install --upgrade pip
#install requirements
python3 -m pip install -r requirements.txt
