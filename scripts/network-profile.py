#!/usr/bin/env python3
"""Save a NetworkManager profile ahead of a move; never disconnect a device."""
import argparse
import configparser
import getpass
import io
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid


def escaped(value):
    # NetworkManager keyfiles use GLib string escaping, not shell quoting.
    return value.replace('\\','\\\\').replace('\n','\\n').replace('\r','\\r').replace('\t','\\t').replace(' ','\\s')


def profile(name, kind, ssid=None, password=None, security='wpa-psk'):
    if not name or any(ord(c)<32 for c in name):raise ValueError('Invalid profile name')
    identity=str(uuid.uuid5(uuid.NAMESPACE_URL,'rgb-network-profile:'+name))
    data=configparser.ConfigParser(interpolation=None)
    data['connection']={'id':escaped(name),'uuid':identity,'type':'wifi' if kind=='wifi' else 'ethernet','autoconnect':'true','autoconnect-retries':'0'}
    metric='600' if kind=='wifi' else '100'
    for version in ['ipv4','ipv6']:data[version]={'method':'auto','route-metric':metric}
    if kind=='wifi':
        if not ssid or len(ssid.encode())>32 or any(ord(c)<32 for c in ssid):raise ValueError('SSID must be 1-32 UTF-8 bytes without control characters')
        if not password or not (8<=len(password.encode())<=63 or re.fullmatch(r'[a-fA-F0-9]{64}',password)):
            raise ValueError('Expected an 8-63 byte passphrase or 64 hex PSK')
        if security not in {'wpa-psk','sae'}:raise ValueError('Invalid security')
        data['wifi']={'mode':'infrastructure','ssid':escaped(ssid)}
        data['wifi-security']={'key-mgmt':security,'psk':escaped(password),'psk-flags':'0'}
    else:data['ethernet']={}
    output=io.StringIO()
    data.write(output,space_around_delimiters=False)
    return identity,output.getvalue()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['wifi','ethernet'])
    parser.add_argument('--name',required=True)
    parser.add_argument('--ssid')
    parser.add_argument('--security',choices=['wpa-psk','sae'],default='wpa-psk')
    parser.add_argument('--replace',action='store_true',help='Replace this tool’s profile with the same name')
    args=parser.parse_args()
    if os.geteuid()!=0:parser.error('Run on the Uno Q with sudo')
    secret=getpass.getpass('Wi-Fi passphrase (stored locally): ') if args.kind=='wifi' else None
    try:identity,contents=profile(args.name,args.kind,args.ssid,secret,args.security)
    except ValueError as exc:parser.error(str(exc))
    directory=Path('/etc/NetworkManager/system-connections')
    path=directory/('rgb-'+identity+'.nmconnection')
    if path.exists() and not args.replace:parser.error('Profile exists; use --replace to update it')
    fd,temporary=tempfile.mkstemp(prefix='.rgb-',dir=directory)
    try:
        with os.fdopen(fd,'w') as stream:stream.write(contents)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    subprocess.run(['nmcli','connection','load',str(path)],check=True,stdout=subprocess.DEVNULL)
    print('Saved profile '+args.name+' ('+identity+'). No connection switch requested.')


if __name__=='__main__':main()
