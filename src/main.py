"""
Main entry point for Knowledge Assistant MCP Server.

This module provides the main() function and server initialization.
"""

import asyncio

from mcp.server.stdio import stdio_server

from .logging import configure_logging
from .tools import server


def main():
    """Main entry point."""
    configure_logging()

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
