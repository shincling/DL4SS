echo "Start to run the python file"
python main_run.py > /dev/null&
touch $!".pid"