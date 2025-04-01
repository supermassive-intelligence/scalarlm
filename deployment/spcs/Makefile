# makefile for the docker images


#
# constants
#
#SF_ACCOUNT="ndsoebe-rai-integration-aws-uswest-1-consumer.registry.snowflakecomputing.com"
#SF_ACCOUNT="wcwhvmh-fmb70117.registry.snowflakecomputing.com"
#SF_ACCOUNT="wcwhvmh-fmb70117.registry.snowflakecomputing.com"
SF_ACCOUNT="ndsoebe-rai-prod-gen-ai-aws-us-west-2-consumer.registry.snowflakecomputing.com"
#SF_DB="graphrag_db"
#SF_SCHEMA="graphrag_schema"
#SF_REPOSITORY="graphrag_repository"
SF_DB="smi_db"
SF_SCHEMA="smi_schema"
SF_REPOSITORY="smi_repo"

#SERVICE_IMAGE="graphrag_image"
SERVICE_IMAGE="cray_service"
#DOCKERFILE="./Dockerfile"
PLATFORM="linux/amd64"
IMAGE_NAME="gdiamos/cray-nvidia:latest"
DOCKER=docker

# for pretty prints
RED="\033[0;31m"
GREEN="\033[0;32m"
NC="\033[0m"

LINE=@printf '%.s─' $$(seq 1 $$(tput cols))
CHECK=@if [ $$?  -eq 0 ]; then echo ${GREEN}[ SUCCESS ]${NC} ; else echo ${RED}[ ERROR ]${NC} ; fi


# don't build, pull an existing image
all: login tag push

#
# logging in
#
login:
	${LINE}
	@echo "Logging into Snowflake ..."
	${LINE}
	${DOCKER} login ${SF_ACCOUNT} 
	${CHECK}

#
# log out
#
logout:
	${DOCKER} logout ${SF_ACCOUNT}


#
# build image
#
#build:
#	${LINE}
#	@echo "Building the docker image ..."
#	${LINE}
#	${DOCKER} build --rm --platform ${PLATFORM} \
#    	-f ${DOCKERFILE} \
#    	-t ${SF_ACCOUNT}/${SF_DB}/${SF_SCHEMA}/${SF_REPOSITORY}/${SERVICE_IMAGE}:latest . 
#	${CHECK}


# pull
pull:
	${LINE}
    ${DOCKER} pull ${IMAGE_NAME}    	
	${CHECK}
#tag image

tag:
	${LINE}
	@echo "Tagging the docker image ..."
	${LINE}
	${DOCKER} tag gdiamos/cray-nvidia:latest ${SF_ACCOUNT}/${SF_DB}/${SF_SCHEMA}/${SF_REPOSITORY}/cray-nvidia:latest
    	
	${CHECK}



#
# push image
#
push:
	${LINE}
	@echo "Pushing the docker image ..."
	${LINE}
	${DOCKER} push \
    	${SF_ACCOUNT}/${SF_DB}/${SF_SCHEMA}/${SF_REPOSITORY}/cray-nvidia:latest
	${CHECK}



#
# cleanup
#
clean:
	@echo "Cleanup"
	rm -rf ./GRAPHRAG_DATA/*


#
# local run
#
run:
	@echo "Running locally ..."
	${DOCKER} run -p 8000:8000 -it ${SERVICE_IMAGE}:latest
	#python graphrag_service.py

#
# stop all local
#
stop:
	@echo "Stopping all local docker containers ..."
	${DOCKER} stop $$(${DOCKER} ps -a -q)

# 
# remove all docker containers
#
rm:
	@echo "Removing all docker containers ..."
	${DOCKER} rm $$(${DOCKER} ps -a -q)


#
# remove all docker images
#
rmi:
	@echo "Removing all docker images ..."
	${DOCKER} rmi $$(${DOCKER} images -q)






