import midtransclient
import os

from dotenv import load_dotenv

dir_path = os.path.dirname(os.path.realpath(__file__))
load_dotenv(dir_path+'/.env')

core_api = midtransclient.CoreApi(
    is_production=False,
    server_key=os.getenv('MIDTRANS_SERVER_KEY'),
    client_key=os.getenv('MIDTRANS_CLIENT_KEY')
)

def execute_payment(data):
    order_id = data["order_id"]
    total = data["total"]
    callback = data["callback_url"]

    param = {
        "payment_type":"gopay",
        "transaction_details": {
            "order_id": order_id,
            "gross_amount": total
        }, 
          "gopay": {
            "enable_callback": True,
            "callback_url": callback
        }
    }

    response = core_api.charge(param)

    return response