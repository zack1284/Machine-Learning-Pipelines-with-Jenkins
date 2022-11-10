
`sudo -S docker build -t adult-model .`

`sudo -S docker run -d --name model adult-model`

`sudo -S docker container exec model python3 preprocessing.py`

`sudo -S docker container exec model python3 train.py`

`sudo -S docker container exec model python3 test.py`

`docker run -p 8080:8080 -p 50000:50000 -d -v jenkins_home:/var/jenkins_home jenkins/jenkins:latest`