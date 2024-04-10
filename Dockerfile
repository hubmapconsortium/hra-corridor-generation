FROM ubuntu:20.04 AS build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN apt install -y build-essential cmake libssl-dev libboost-all-dev libgmp-dev libmpfr-dev libeigen3-dev libassimp-dev libcpprest-dev
RUN apt install -y wget unzip
WORKDIR /usr/src/app/build
COPY corridor_http_service ..
#RUN mkdir /tmp
RUN wget https://github.com/CGAL/cgal/releases/download/v5.5.3/CGAL-5.5.3.zip
RUN unzip CGAL-5.5.3.zip -d /usr/src/app
ENV HOME /usr/src/app
RUN rm CGAL-5.5.3.zip
RUN cmake .. 
RUN make

FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y libssl-dev libboost-all-dev libgmp-dev libmpfr-dev libeigen3-dev libassimp-dev libcpprest-dev
WORKDIR /usr/src/app
COPY --from=build /usr/src/app/build/server2 .
COPY model model
COPY server.sh .

ENV PORT 8080
EXPOSE 8080
RUN chmod +x server.sh
CMD [ "./server2", "model/organ_origins_meter_v1.4.csv", "model/asct-b-3d-models-crosswalk.csv", "model/plain_manifold_filling_hole_v1.4/", "model/reference-organ-data.json", "0.0.0.0", "8080"]
