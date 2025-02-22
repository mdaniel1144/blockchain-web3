import json
import ipfshttpclient
from neural_network import deserialize_gradients


#--> Connect to IPFS server
def connect_to_ipfs() -> ipfshttpclient.client.Client:
    try:
        client = ipfshttpclient.connect('/dns/localhost/tcp/5001/http')
        return client
    except ipfshttpclient.exceptions.ConnectionError as e:
        raise


#--> Get Gradients, store it in IPFS and Return the hash code
def IPFS_Store(info):
    client = connect_to_ipfs()
    try:
        res = client.add_json(info)
        hash_value = res
        print(f"    Successfully stored data in IPFS with hash: {hash_value}")
        return hash_value
    
    except Exception as e:
        print(f"    Error storing data in IPFS: {e}")
        raise
    finally:
        client.close()


#--> Get Hash code and return the gradeint as Dict
def IPFS_ExtractStore(hash_info: str):
    client = connect_to_ipfs()
    try:
        retrieved_data = client.cat(hash_info)
        decoded_data = json.loads(retrieved_data.decode('utf-8'))
        return deserialize_gradients(decoded_data)
    except Exception as e:
        print(f"    Error retrieving data from IPFS: {e}")
        raise
    finally:
        client.close()