import streamlit as st
import argparse
import importlib
import os
from streamlit.logger import get_logger

logger = get_logger(__name__)


def to_label(s):
    return ' '.join([w[0].upper() + w[1:] for w in s.split('_') if w.isalnum()])


def main():
    st.set_page_config(
        page_title='MMD Toolbox',
        page_icon='🛠️',
        layout='wide',
    )
    st.page_link('🏠_Home.py', label='Home', icon='🏠')
    for f in os.listdir('pages'):
        if f.endswith('.py'):
            st.page_link(f'pages/{f}', label=to_label(f[:-3]), icon=f[0])


if __name__ == '__main__':
    main()
