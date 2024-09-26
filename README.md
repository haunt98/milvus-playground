# milvus

https://milvus.io/docs/install_standalone-docker.md

https://milvus.io/docs/integrate_with_hugging-face.md

```sh
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Make a copy standalone_embed_no_sudo.sh without sudo
bash standalone_embed_no_sudo.sh start
```

Dependencies:

```sh
pip install -U pymilvus
pip install -U "pymilvus[model]"
pip install transformers torch accelerate
```
