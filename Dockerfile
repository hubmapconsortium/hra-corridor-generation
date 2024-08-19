
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 AS build
#FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN apt install -y build-essential cmake libssl-dev libboost-all-dev libgmp-dev libmpfr-dev libeigen3-dev libassimp-dev libcpprest-dev gcc-10 g++-10
RUN apt install -y wget unzip
WORKDIR /usr/src/app/build
COPY src ..

#RUN mkdir /tmp
RUN wget https://github.com/CGAL/cgal/releases/download/v5.5.3/CGAL-5.5.3.zip
RUN unzip CGAL-5.5.3.zip -d /usr/src/app
ENV HOME /usr/src/app
RUN rm CGAL-5.5.3.zip
RUN cmake .. 
RUN make

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
#FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y libssl-dev libboost-all-dev libgmp-dev libmpfr-dev libeigen3-dev libassimp-dev libcpprest-dev
WORKDIR /usr/src/app
COPY --from=build /usr/src/app/build/corridor_api .
COPY model model
COPY server.sh .

ENV PORT 8080
EXPOSE 8080
ENV CPU_GPU="CPU"
RUN chmod +x server.sh
CMD [ "./server.sh" ]
