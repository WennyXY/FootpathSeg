import geopandas
from shapely.geometry import Polygon
from shapely.ops import unary_union, cascaded_union
import pandas as pd
import os
import math
import cv2
import argparse
from get_geoinfo_from_name import num2deg
from tile2net_utils.geodata_utils import *
from tile2net_utils.topology import *
import utils


def img2gis(img_path, output_path, data_path= None, gdf_file=None, create_line=True, tolerance=10, min_area=10):
    # 
    # input: imgs, output path
    # save a geojson file
    # 
    if gdf_file:
        print('..... load gdf file')
        gdf = geopandas.read_file(gdf_file)
        gdf = gdf.to_crs(crs = 4326)
    else:
        print('..... from masks to gdf')
        if data_path:
            with open(data_path, "r") as f:
                imgs = f.readlines()
                if len(imgs) == 1:
                    imgs = imgs[0].strip().split(' ')
        else:
            imgs = os.listdir(img_path)
        print('img_path', img_path, len(imgs), 'output_path',output_path)
        i = 0
        polygon_list = []
        for img_name in imgs:
            img_name = img_name.strip()
            i += 1
            img = cv2.imread(img_path+img_name)
            width, height = img.shape[:2]
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            polygon_tmp = utils.img_coord2lat_lon(img_name, width, height, contours)
            # break
            for poly in polygon_tmp:
                if len(poly) > 3:
                    polygon_list.append(Polygon(poly))
        gdf = geopandas.GeoDataFrame(geometry = polygon_list, crs="EPSG:4326")

    
    # get the union parts
    gdf = gdf.to_crs(crs=3857)
    
    # preprocess refer to tile2net
    gdf = preprocess_pdf(gdf, 3857, min_area)
    gdf.to_file(output_path+'_simplify_filter.geojson', driver='GeoJSON')
    print('simplified GIS file saved')
    
    if create_line:
        print('..... creating the centerlines network')
        # create lines
        lines = create_lines(gdf, tolerance=tolerance)
        
        print('..... processing the centerlines network')
        modif_uni = geopandas.GeoDataFrame(
                    geometry=geopandas.GeoSeries([geom for geom in lines.unary_union.geoms]))
        modif_uni_met = set_gdf_crs(modif_uni, 3857)
        uni_lines = modif_uni_met.explode(index_parts=True)
        uni_lines.reset_index(drop=True, inplace=True)
        uni_lines.dropna(inplace=True)
        
        centerline_network = uni_lines
        centerline_network = change_crs(centerline_network, 4326)
        
        print('..... saving the centerlines')
        centerline_network.to_file(output_path+str(tolerance)+'_centerline.geojson', driver='GeoJSON')
        # print('polygon_list:', len(polygon_list))
        print('generate centerline network successfully')

def preprocess_pdf(gdf, crs_metric=3857, min_area=10):
    gdf.geometry = gdf.simplify(0.6)
    # union and buffer the polygons of each class separately,
	# to create continuous polygons and merge them into one GeoDataFrame.
    unioned = buff_dfs(gdf)
    unioned.geometry = unioned.geometry.simplify(0.9)
    unioned.dropna(inplace=True)
    unioned = unioned[unioned.geometry.notna()]
    # finds holes in the polygons
    # dataframe.apply: Apply a function along an axis of the DataFrame.
    unioned['geometry'] = unioned.apply(fill_holes, args=(25,), axis=1)
    # replace the convex polygons with their envelopes
    simplified = replace_convexhull(unioned)
    
    simplified = geopandas.GeoDataFrame(
                    geometry=geopandas.GeoSeries([geom for geom in simplified.geometry if geom.area > min_area]))
    
    simplified.set_crs(crs_metric, inplace=True)
    
    return simplified


def create_lines(gdf: geopandas.GeoDataFrame, tolerance=10) -> geopandas.GeoDataFrame:
    """
    from tile2net
    Create centerlines from polygons
    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        geodataframes of polygons
    -------

    """
    lin_geom = []
    gdf_atts = morpho_atts(gdf)
    for c, geom in enumerate(gdf.geometry):
        corners = gdf_atts.iloc[c, gdf_atts.columns.get_loc(gdf_atts.corners.name)]
        minpr = 2 * math.sqrt(math.pi * abs(geom.area))
        trim1 = 20
        trim2 = 6
        if geom.area <= 20:  # 45 DC #10 others
            continue
        elif minpr / geom.length > 0.8:
            # it is close to a circle
            # the interpolation distance to be 2/3rd of the minimum circle perimeter
            cl_arg = math.sqrt(geom.area / math.pi) / 4
        else:
            cl_arg = 0.2

        line = to_cline(geom, cl_arg, 1)
        if not line.is_empty:
            tr_line_ = trim_checkempty(line, trim1, trim2)
            if corners > 100:
                tr_line = trim_checkempty(tr_line_, trim1, trim2)
            else:
                tr_line = tr_line_

        else:
            line_clh = to_cline(geom, cl_arg / 2, 1)
            if not line_clh.is_empty:
                tr_line_ = trim_checkempty(line_clh, trim1, trim2)
                if corners > 100:
                    tr_line = trim_checkempty(tr_line_, trim1, trim2)
                else:
                    tr_line = tr_line_
            else:
                new_line = to_cline(geom, 0.1, 0.5)
                tr_line_ = trim_checkempty(new_line, trim1, trim2)
                if corners > 100:
                    tr_line = trim_checkempty(tr_line_, trim1, trim2)
                else:
                    tr_line = tr_line_
        if tr_line.is_empty:
            print('empty line')
            continue
        else:
            line_tr = tr_line.simplify(1)
            extended_line = extend_lines(geo2geodf([line_tr]),
                                            target=geo2geodf([geom.boundary]), tolerance=tolerance, extension=0)
            lin_geom.append(extended_line)

    if len(lin_geom) > 0:
        ntw = pd.concat(lin_geom)
        smoothed = wrinkle_remover(ntw, 1.5)
        return smoothed

def connect_lines(gdf):
    points = get_line_sepoints(gdf)
    
    # query LineString geometry to identify points intersecting 2 geometries
    inp, res = gdf.sindex.query(geo2geodf(points).geometry,
                                                predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    ends = np.unique(res[np.isin(inp, unique[counts == 1])])
    
    new_geoms_s = []
    new_geoms_e = []
    new_geoms_both = []
    all_connections = []
    # iterate over crosswalk segments that are not connected to other crosswalk segments
    # and add the start and end points to the new_geoms
    pgeom = gdf.geometry.values
    for line in ends:
        l_coords = shapely.get_coordinates(pgeom[line])

        start = Point(l_coords[0])
        end = Point(l_coords[-1])

        first = list(pgeom.sindex.query(start, predicate="intersects"))
        second = list(pgeom.sindex.query(end, predicate="intersects"))
        first.remove(line)
        second.remove(line)

        if first and not second:
            new_geoms_s.append((line, end))

        elif not first and second:
            new_geoms_e.append((line, start))
        if not first and not second:
            new_geoms_both.append((line, start))
            new_geoms_both.append((line, end))
    # create a dataframe of points
    if len(new_geoms_s) > 0:
        ps = [g[1] for g in new_geoms_s]
        ls = [g[0] for g in new_geoms_s]
        pdfs = gpd.GeoDataFrame(geometry=ps)
        pdfs.set_crs(3857, inplace=True)

        connect_s = get_shortest(gdf, pdfs, f_type='sidewalk_connection')
        all_connections.append(connect_s)

    if len(new_geoms_e) > 0:
        pe = [g[1] for g in new_geoms_e]
        le = [g[0] for g in new_geoms_e]
        pdfe = gpd.GeoDataFrame(geometry=pe)
        pdfe.set_crs(3857, inplace=True)

        connect_e = get_shortest(gdf, pdfe, f_type='sidewalk_connection')
        all_connections.append(connect_e)

    if len(new_geoms_both) > 0:
        pb = [g[1] for g in new_geoms_both]
        lb = [g[0] for g in new_geoms_both]  # crosswalk lines where both ends do not intersect
        pdfb = gpd.GeoDataFrame(geometry=pb)
        pdfb.set_crs(3857, inplace=True)

        connect_b = get_shortest(gdf, pdfb, f_type='sidewalk_connection')
        all_connections.append(connect_b)
    
    if len(all_connections) > 1:
        connect = pd.concat(all_connections)
    elif len(all_connections) == 1:
        connect = all_connections[0]
    else:
        connect = []

    if len(all_connections) > 0:
        combined = pd.concat([gdf, connect])
    else:
        print('......No combination......')
        combined = gdf

    combined.dropna(inplace=True)
    combined.geometry = combined.geometry.set_crs(3857)
    combined.geometry = combined.geometry.to_crs(4326)
    combined = combined[~combined.geometry.isna()]
    combined.reset_index(drop=True, inplace=True)

    return combined

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 
    parser.add_argument('--img_path', default='', type=str, help="Original img path.")
    parser.add_argument('--output_path', default='', type=str, help="Output path.")
    parser.add_argument('--data_path', default=None, type=str, help="Output path.")
    parser.add_argument('--gdf_file', default=None, type=str, help="gdf file path.")
    parser.add_argument('--create_line', default=False, action='store_true', help="whether extract centerline.")
    parser.add_argument('--tolerance', default=10, type=int, help="tolerance for extending lines.")
    parser.add_argument('--min_area', default=10, type=int, help="min area to filter the polygons.")

    args = parser.parse_args()
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
   
    img2gis(args.img_path, args.output_path, args.data_path, gdf_file = args.gdf_file, create_line = args.create_line, tolerance = args.tolerance, min_area=args.min_area)
    # root = '/data/scratch/projects/punim1671/xinye/outputs/wr101_mc/cycle_1k/inter_7865/bikenetwork'
    # gdf_file = root + '4_simplify_filter.geojson'
    # gdf_new = geopandas.read_file(gdf_file)
    # for i in range(1,4):
    #     gdf_file = root + str(i) + '_simplify_filter.geojson'
    #     print(gdf_file)
    #     gdf = geopandas.read_file(gdf_file)
    #     gdf_new = pd.concat([gdf, gdf_new])
    # gdf_new.to_file(root+'_combine.geojson', driver='GeoJSON')
    
    # gdf_combine = geopandas.read_file(root+'_combine_new.geojson')
    
    # gdf_combine = gdf_combine.to_crs(crs=3857)
    # simplified = geopandas.GeoDataFrame(
    #                 geometry=geopandas.GeoSeries([geom for geom in gdf_combine.geometry if geom.area > 10]),crs=3857)
    # # simplified.set_crs(3857, inplace=True)
    # simplified = simplified.to_crs(4326)
    # simplified.to_file(root+'_combine_simplified.geojson', driver='GeoJSON')