def wind_chill(v, t):
	"""Return rounded wind chill in Celsius for wind speed v in kilometers per hour and temperature t."""
	
	wind_factor = v ** 0.16
	return round(13.12 + 0.6215 * t - 11.37 * wind_factor + 0.3965 * t * wind_factor)
