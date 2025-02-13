set -x
echo "mounting: $1"

if [ ! -f "/c/Users/v-zezhongjin/Desktop/MSRA_intern/inkai.prototype-test/blobs/$1.yaml" ]
then
    echo "/c/Users/v-zezhongjin/Desktop/MSRA_intern/inkai.prototype-test/blobs/$1.yaml does not exit!"
    exit 1
fi


if [ ! -d "/c/Users/v-zezhongjin/Desktop/MSRA_intern/inkai.prototype-test/blobs/$1" ]
then
    mkdir -p "/c/Users/v-zezhongjin/Desktop/MSRA_intern/inkai.prototype-test/blobs/$1"
fi

if [ ! -d "/c/Users/v-zezhongjin/Desktop/MSRA_intern/inkai.prototype-test/blobs/$1_tmp" ]
then
    mkdir -p "/c/Users/v-zezhongjin/Desktop/MSRA_intern/inkai.prototype-test/blobs/$1_tmp"
fi


blobfuse2 mount /c/Users/v-zezhongjin/Desktop/MSRA_intern/inkai.prototype-test/blobs/$1  --config-file=/c/Users/v-zezhongjin/Desktop/MSRA_intern/inkai.prototype-test/blobs/$1.yaml 

echo "$1 mounted!"
