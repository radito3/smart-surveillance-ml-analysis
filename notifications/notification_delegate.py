import logging
import time
import requests


def send_notification(address: str) -> None:
    retry_attempts: int = 3
    while retry_attempts > 0:
        ok, error = _send_notification_internal(address)
        if ok:
            return
        logging.debug(f"Sending notification failed: {error}. Retrying after 2 seconds")
        retry_attempts -= 1
        time.sleep(2)
    logging.error(f"Could not send notification within 3 attempts. Consider investigating the issue")


def _send_notification_internal(address: str) -> tuple[bool, str]:
    logging.info(f"Sending notification to: {address}")
    response = requests.post(address)
    if response.status_code != 200:
        return False, response.text
    else:
        logging.info('Notification sent')
        return True, ''
