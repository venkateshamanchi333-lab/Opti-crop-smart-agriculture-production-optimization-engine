import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app


def test_status_endpoint():
    client = app.test_client()
    response = client.get('/api/status')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'ok'


def test_home_page_renders():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200


def test_compare_page_renders():
    client = app.test_client()
    response = client.get('/compare')
    assert response.status_code == 200
