import math

def num2deg(xtile, ytile, zoom):
    # 
    # tiles number to lat & lon
    # 
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def get_info_from_name(img_name):
    lon_lat, coordinates, _, zoom  = img_name.split('~')
    lon, lat = lon_lat.split(',')
    lon = float(lon)
    lat = float(lat)
    x_tile, y_tile = coordinates.split(',')
    zoom = int(zoom.split('.')[0])
    
    next_tile_lat, next_tile_lon = num2deg(int(x_tile)+1, int(y_tile)+1, zoom)
    
    min_lon = min(lon, next_tile_lon)
    min_lat = min(lat, next_tile_lat)
    max_lon = max(lon, next_tile_lon)
    max_lat = max(lat, next_tile_lat)
    return lon, lat, next_tile_lon, next_tile_lat

def img_coord2lat_lon(img_name, width, height, coord_list):
    lon, lat, next_tile_lon, next_tile_lat = get_info_from_name(img_name)
    lon_dist = next_tile_lon - lon
    lat_dist = next_tile_lat - lat
    new_list = []
    for polygon in coord_list:
        points = []
        for point in polygon:
            # print(point)
            p_lat = point[0][1] * lat_dist / width
            p_lon = point[0][0] * lon_dist / height
            points.append((lon+p_lon, lat+p_lat))
        if len(points) > 0:
            new_list.append(points)
    
    return new_list