## notes on setting up debian server

(will need to set all this up in a docker or other container when creating fully functioning version
(debian 9))

get some packages

`sudo apt-get update && sudo apt-get install git python3-pip nginx`

install pip packages

`pip3 install numpy pandas pydub librosa tensorflow keras tqdm h5py jupyter`

generate template jupyter config

`jupyter notebook --generate-config`

set jupyter settings (note: make sure to make internal IP static)
```
ip = '*'
port=8888
openBrowser = False
running ip <internal_ip>
```

nginx proxy port 80 -> 8888 so can access from network with non-standard ports locked down (e.g. work). Add following to nginx.conf
```
upstream notebook {
    server <internal_ip>:8888;
}
server{
listen 80;
server_name xyz.abc.com;
location / {
        proxy_pass            http://notebook;
        proxy_set_header      Host $host;
}
location ~ /api/kernels/ {
        proxy_pass            http://notebook;
        proxy_set_header      Host $host;
        # websocket support
        proxy_http_version    1.1;
        proxy_set_header      Upgrade "websocket";
        proxy_set_header      Connection "Upgrade";
        proxy_read_timeout    86400;
}
location ~ /terminals/ {
        proxy_pass            http://notebook;
        proxy_set_header      Host $host;
        # websocket support
        proxy_http_version    1.1;
        proxy_set_header      Upgrade "websocket";
        proxy_set_header      Connection "Upgrade";
        proxy_read_timeout    86400;
}
}
```


start nginx
```
nginx
nginx -s reload
```

start jupyter notebook
`jupyter notebook`