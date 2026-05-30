"""
servers/weather_server.py
==========================
A minimal MCP weather server using FastMCP with streamable-HTTP transport.
Run as a server: python servers/weather_server.py
Default port: 8000  →  http://localhost:8000/mcp

This server exposes weather tools used by HTTP transport examples.
"""

from fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(location: str) -> str:
    """
    Get the current weather for a location.

    Args:
        location: City name or location string.
    """
    # Simulated weather data
    weather_data = {
        "new york":  "Partly cloudy, 18°C, humidity 65%, wind 12 km/h",
        "london":    "Overcast, 14°C, humidity 80%, wind 20 km/h",
        "tokyo":     "Sunny, 22°C, humidity 55%, wind 8 km/h",
        "paris":     "Rainy, 12°C, humidity 90%, wind 15 km/h",
        "sydney":    "Clear, 26°C, humidity 40%, wind 10 km/h",
    }
    key = location.lower().strip()
    return weather_data.get(key, f"Sunny and warm in {location}! (simulated data)")


@mcp.tool()
async def get_forecast(location: str, days: int = 3) -> str:
    """
    Get a multi-day weather forecast.

    Args:
        location: City name or location string.
        days:     Number of forecast days (1-7).
    """
    days = max(1, min(days, 7))
    return (
        f"Weather forecast for {location} ({days} days):\n"
        + "\n".join(f"  Day {i+1}: Partly cloudy, {18+i}°C" for i in range(days))
    )


@mcp.tool()
async def get_air_quality(location: str) -> dict:
    """
    Get air quality index for a location.

    Args:
        location: City name or location string.
    """
    return {
        "location": location,
        "aqi":      42,
        "category": "Good",
        "pm25":     8.5,
        "pm10":     20.1,
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)
