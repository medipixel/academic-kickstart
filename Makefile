init:
	git submodule init
	git submodule update --remote

deploy:
	chmod +x deploy.sh
	sh deploy.sh
