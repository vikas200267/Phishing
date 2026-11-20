from __future__ import annotations
import ipaddress, socket, ssl
from urllib.parse import urlparse

def _public_host(host: str) -> bool:
    try:
        ip=ipaddress.ip_address(host)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified)
    except ValueError:
        return bool(host)

def _public_resolved_ips(host: str, port: int=443) -> list[str]:
    if not host or not _public_host(host): return []
    infos=socket.getaddrinfo(host,port,type=socket.SOCK_STREAM)
    ips=[]
    for x in infos:
        ip=x[4][0]
        if _public_host(ip): ips.append(ip)
    return sorted(set(ips))

def resolve_host(host: str) -> dict:
    if not host or not _public_host(host): return {'ok':False,'reason':'non_public_host'}
    try:
        infos=socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)
        all_ips=sorted({x[4][0] for x in infos})
        public_ips=[x for x in all_ips if _public_host(x)]
        if not public_ips: return {'ok':False,'reason':'private_or_local_resolution','resolved_ips':all_ips[:20]}
        return {'ok':True,'ips':public_ips[:20],'count':len(public_ips),'resolved_total':len(all_ips)}
    except Exception as e:
        return {'ok':False,'error':type(e).__name__}

def tls_summary(url: str, timeout: float=2.0) -> dict:
    p=urlparse(url); host=p.hostname or ''
    if p.scheme!='https' or not host or not _public_host(host): return {'ok':False,'reason':'https_public_host_required'}
    try:
        port=p.port or 443
        if not 1<=port<=65535: return {'ok':False,'reason':'invalid_port'}
        ips=_public_resolved_ips(host,port)
        if not ips: return {'ok':False,'reason':'private_or_local_resolution'}
        target_ip=ips[0]
    except (ValueError,OSError) as e:
        return {'ok':False,'error':type(e).__name__}
    ctx=ssl.create_default_context()
    try:
        with socket.create_connection((target_ip,port),timeout=timeout) as sock:
            with ctx.wrap_socket(sock,server_hostname=host) as ssock:
                cert=ssock.getpeercert()
                return {'ok':True,'resolved_ip':target_ip,'tls_version':ssock.version(),'cipher':ssock.cipher()[0] if ssock.cipher() else None,'subject':cert.get('subject',[]),'issuer':cert.get('issuer',[]),'not_after':cert.get('notAfter')}
    except Exception as e:
        return {'ok':False,'error':type(e).__name__}
