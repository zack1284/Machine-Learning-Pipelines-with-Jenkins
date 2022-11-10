
## ML Pipeline with Jenkins

## Change Ip Address

chang the IP address in 'test.py'

## NROK

The github webhook would not be able to work if you use IP Address, it needs to be smth like http://xxx.com.
Therefore, we will need to use NROK to assign a domain to us so that we can link github and jenkins together.
However, the webhook will expire in certain period of time. So you need to reassign.

## Note

In Jenkins's pipeline configuration, we use scripts to build our pipelines. We need to remove models after building one, so that we can keep using the same model name next time.

## Some commands

`sudo -S docker build -t adult-model .`

`sudo -S docker run -it -d --name model adult-model`

`sudo -S docker container exec model python3 test.py`

`docker run -p 8080:8080 -p 50000:50000 -d -v jenkins_home:/var/jenkins_home jenkins/jenkins:latest`