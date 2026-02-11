"""FastMCP server for ALFWorld environment."""

from fastmcp import FastMCP

from src.environment.env_manager import EnvManager

# Create FastMCP server instance
mcp = FastMCP(name="ALFWorldServer")

# Create environment manager (load will be called by the caller when ready)
env_manager = EnvManager()
