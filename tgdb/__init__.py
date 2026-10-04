import config as C

def get_tg():
    """TG_BACKEND=mcp -> agent/vector calls go through the TigerGraph MCP server; rest -> pyTigerGraph only."""
    if C.TG_BACKEND == "mcp":
        from tgdb.mcp_client import MCPTG
        return MCPTG()
    from tgdb.client import TG
    return TG()
