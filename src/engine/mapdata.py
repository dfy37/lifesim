import os
import pickle
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import pyproj
import shapely
from google.protobuf.json_format import MessageToDict
from pycityproto.city.map.v2 import map_pb2
from pydantic import BaseModel, Field
from shapely.geometry import Point, Polygon

from engine.map_const import POI_CATG_DICT
from logger import get_logger

__all__ = ["MapData", "MapConfig"]


class MapConfig(BaseModel):
    """Map configuration class."""

    file_path: str = Field(...)
    """Path to the map file. If s3 is enabled, the file will be downloaded from S3"""


class MapData:
    """
    地图API
    Map API
    """

    def __init__(self, config: MapConfig):
        """
        Args:
        - config (MapConfig): Map config, Defaults to None. Map config.
        """
        get_logger().info("MapData init")
        map_data = None
        # 1. try to load from cache
        cache_path = config.file_path + '.cache'
        exists = os.path.exists(cache_path)
        if exists:
            get_logger().info("Start load cache file in MapData")
            with open(cache_path, "rb") as f:
                map_data = pickle.load(f)
            get_logger().info("Finish load cache file in MapData")
        if map_data is None:
            get_logger().info("No cache file found, start parse pb file in MapData")
            with open(config.file_path, "rb") as f:
                pb = map_pb2.Map().FromString(f.read())

            jsons = []
            # add header
            jsons.append(
                {
                    "class": "header",
                    "data": MessageToDict(
                        pb.header,
                        including_default_value_fields=True,
                        preserving_proto_field_name=True,
                        use_integers_for_enums=True,
                    ),
                }
            )
            # add aois
            for aoi in pb.aois:
                jsons.append(
                    {
                        "class": "aoi",
                        "data": MessageToDict(
                            aoi,
                            including_default_value_fields=True,
                            preserving_proto_field_name=True,
                            use_integers_for_enums=True,
                        ),
                    }
                )
            # add pois
            for poi in pb.pois:
                jsons.append(
                    {
                        "class": "poi",
                        "data": MessageToDict(
                            poi,
                            including_default_value_fields=True,
                            preserving_proto_field_name=True,
                            use_integers_for_enums=True,
                        ),
                    }
                )
            map_data = self._parse_map(jsons)
            get_logger().info("Finish parse pb file")
            if not exists:
                get_logger().info("Start save cache file")
                with open(cache_path, "wb") as f:
                    pickle.dump(map_data, f)
                get_logger().info("Finish save cache file")

        self.header: dict = map_data["header"]
        """
        地图元数据，包含如下属性:
        Map metadata, including the following attributes:
        - name (string): 城市道路名称，供标识数据集合的语义。Map name, to identify the semantics of data collections.
        - date (string): 城市道路数据的创建时间。Map data creation time.
        - north (float): 道路数据的北边界坐标。The coordinate of the northern boundary of the Map data.
        - south (float): 道路数据的南边界坐标。The coordinate of the southern boundary of the Map data.
        - east (float): 道路数据的东边界坐标。The coordinate of the eastern boundary of the Map data.
        - west (float): 道路数据的西边界坐标。The coordinate of the western boundary of the Map data.
        - projection (string): PROJ.4 投影字符串，用以支持xy坐标到其他坐标系的转换。PROJ.4 projection string to support the conversion of xy coordinates to other coordinate systems.
        """

        self.aois: Dict[int, dict] = map_data["aois"]
        """
        地图中的AOI集合（aoi），字典的值包含如下属性:
        AOI collection (aoi) in the map, the value of the dictionary contains the following attributes:
        - id (int): AOI编号。AOI ID.
        - positions (list[XYPosition]): 多边形空间范围。Shape of polygon.
        - area (float): 面积(单位: m2)。Area.
        - driving_positions (list[LanePosition]): 和道路网中行车道的连接点。Connection points to driving lanes.
        - walking_positions (list[LanePosition]): 和道路网中人行道的连接点。Connection points to pedestrian lanes.
        - driving_gates (list[XYPosition]): 和道路网中行车道的连接点对应的AOI边界上的位置。Position on the AOI boundary corresponding to the connection point to driving lanes.
        - walking_gates (list[XYPosition]): 和道路网中人行道的连接点对应的AOI边界上的位置。Position on the AOI boundary corresponding to the connection point to pedestrian lanes.
        - urban_land_use (Optional[str]): 城市建设用地分类，参照执行标准GB 50137-2011（https://www.planning.org.cn/law/uploads/2013/1383993139.pdf） Urban Land use type, refer to the national standard GB 50137-2011.
        - poi_ids (list[int]): 包含的POI列表。Contained POI IDs.
        - shapely_xy (shapely.geometry.Polygon): AOI的形状（xy坐标系）。Shape of polygon (in xy coordinates).
        - shapely_lnglat (shapely.geometry.Polygon): AOI的形状（经纬度坐标系）。Shape of polygon (in latitude and longitude).
        """

        self.pois: Dict[int, dict] = map_data["pois"]
        """
        地图中的POI集合（poi），字典的值包含如下属性:
        POI collection (poi) in the map, the value of the dictionary contains the following attributes:
        - id (int): POI编号。POI ID.
        - name (string): POI名称。POI name.
        - category (string): POI类别编码。POI category code.
        - position (XYPosition): POI位置。POI position.
        - aoi_id (int): POI所属的AOI编号。AOI ID to which the POI belongs.
        """

        (
            self._aoi_tree,
            self._aoi_list,
            self._poi_tree,
            self._poi_list,
        ) = self._build_geo_index()

        self.poi_cate = POI_CATG_DICT

    def get_all_aois(self):
        return self._aoi_list

    def get_aoi(self, aoi_id: int):
        return self.aois[aoi_id]

    def get_all_pois(self):
        return self._poi_list

    def get_poi(self, poi_id: int):
        return self.pois[poi_id]

    def _parse_map(self, m: List[Any]) -> Dict[str, Any]:
        # client = MongoClient(uri)
        # m = list(client[db][coll].find({}))
        get_logger().info("Start parse map data")
        header = None
        aois = {}
        pois = {}
        for d in m:
            if "_id" in d:
                del d["_id"]
            t = d["class"]
            data = d["data"]
            if t == "aoi":
                aois[data["id"]] = data
            elif t == "poi":
                pois[data["id"]] = data
            elif t == "header":
                header = data
        assert header is not None, "header is None"
        get_logger().info("Finish parse map data - classify")
        projector = pyproj.Proj(header["projection"])  #
        # 处理Aoi的Geos
        get_logger().info("Start process aoi geos")
        for aoi in aois.values():
            if "area" not in aoi:
                # 不是多边形aoi
                aoi["shapely_xy"] = Point(
                    aoi["positions"][0]["x"], aoi["positions"][0]["y"]
                )
            else:
                aoi["shapely_xy"] = Polygon(
                    [(one["x"], one["y"]) for one in aoi["positions"]]
                )
            xys = np.array([[one["x"], one["y"]] for one in aoi["positions"]])
            lngs, lats = projector(xys[:, 0], xys[:, 1], inverse=True)
            lnglat_positions = list(zip(lngs, lats))
            if "area" not in aoi:
                aoi["shapely_lnglat"] = Point(lnglat_positions[0])
            else:
                aoi["shapely_lnglat"] = Polygon(lnglat_positions)
        get_logger().info("Finish process aoi geos in MapData")
        # 处理Poi的Geos
        get_logger().info("Start process poi geos in MapData")
        for poi in pois.values():
            point = Point(poi["position"]["x"], poi["position"]["y"])
            poi["shapely_xy"] = point
            lng, lat = projector(point.x, point.y, inverse=True)
            poi["shapely_lnglat"] = Point([lng, lat])
        get_logger().info("Finish process poi geos")

        return {
            "header": header,
            "aois": aois,
            "pois": pois,
        }

    def _build_geo_index(self):
        # poi:
        # {
        #     "id": 700000000,
        #     "name": "天翼(互联网手机卖场)",
        #     "category": "131300",
        #     "position": {
        #       "x": 448802.148620172,
        #       "y": 4412128.118718166
        #     },
        #     "aoi_id": 500018954,
        # }
        get_logger().info("Start build geo index in MapData")
        aoi_list = list(self.aois.values())
        aoi_tree = shapely.STRtree([aoi["shapely_xy"] for aoi in aoi_list])
        poi_list = list(self.pois.values())
        poi_tree = shapely.STRtree([poi["shapely_xy"] for poi in poi_list])
        get_logger().info("Finish build geo index in MapData")
        return (
            aoi_tree,
            aoi_list,
            poi_tree,
            poi_list,
        )

    @overload
    def query_pois(
        self,
        center: Union[Tuple[float, float], Point],
        radius: Optional[float] = None,
        category_prefix: Optional[str] = None,
        limit: Optional[int] = None,
        return_distance: Literal[False] = False,
    ) -> List[Any]: ...

    @overload
    def query_pois(
        self,
        center: Union[Tuple[float, float], Point],
        radius: Optional[float] = None,
        category_prefix: Optional[str] = None,
        limit: Optional[int] = None,
        return_distance: Literal[True] = True,
    ) -> List[Tuple[Any, float]]: ...

    def query_pois(
        self,
        center: Union[Tuple[float, float], Point],
        radius: Optional[float] = None,
        category_prefix: Optional[str] = None,
        limit: Optional[int] = None,
        return_distance: bool = True,
    ) -> Union[List[Tuple[Any, float]], List[Any]]:
        """
        查询center点指定半径内类别满足前缀的poi（按距离排序）。Query the POIs whose categories satisfy the prefix within the specified radius of the center point (sorted by distance).

        Args:
        - center (x, y): 中心点（xy坐标系）。Center point (xy coordinate system).
        - radius (float, optional): 半径（单位：m）。如果不提供则返回所有的poi。Radius (unit: m).If not provided, all pois within the map will be returned.
        - category_prefix (str, optional): 类别前缀，如实际类别为100000，那么匹配的前缀可以为10、1000等。Category prefix, if the actual category is 100000, then the matching prefix can be 10, 1000, etc.
        - limit (int, optional): 最多返回的poi数量，按距离排序，近的优先（默认None）. The maximum number of POIs returned, sorted by distance, closest ones first (default to None).
        - return_distance (bool): 是否返回距离。Return the distance or not.

        Returns:
        - Union[List[Tuple[Any, float]],List[Any]]: poi列表，每个元素为（poi, 距离）或者poi。poi list, each element is (poi, distance) or poi.
        """
        if not isinstance(center, Point):
            center = Point(center)
        if radius is None:
            if return_distance:
                pois = [(p, center.distance(p["shapely_xy"])) for p in self._poi_list]
            else:
                pois = [p for p in self._poi_list]
        else:
            # 获取半径内的poi
            indices = self._poi_tree.query(center.buffer(radius))
            # 过滤掉不满足类别前缀的poi
            pois = []
            for index in indices:
                poi = self._poi_list[index]
                if category_prefix is None or poi["category"].startswith(
                    category_prefix
                ):
                    if return_distance:
                        distance = center.distance(poi["shapely_xy"])
                        pois.append((poi, distance))
                    else:
                        pois.append(poi)
        if return_distance:
            # 按照距离排序
            pois = sorted(pois, key=lambda x: x[1])
        if limit is not None:
            pois = pois[:limit]
        return pois

    def get_poi_cate(self):
        return self.poi_cate

    def get_map_header(self):
        return self.header

    def get_projector(self):
        return self.header["projection"]

    def query_aois(
        self,
        center: Union[Tuple[float, float], Point],
        radius: float,
        urban_land_uses: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """
        查询center点指定半径内城市用地满足条件的aoi（按距离排序）。Query the AOIs whose urban land use within the specified radius of the center point meets the conditions (sorted by distance).

        Args:
        - center (x, y): 中心点（xy坐标系）。Center point (xy coordinate system).
        - radius (float): 半径（单位：m）。Radius (unit: m).
        - urban_land_uses (List[str], optional): 城市用地分类列表，参照执行标准GB 50137-2011（https://www.planning.org.cn/law/uploads/2013/1383993139.pdf）. Urban land use classification list, refer to the national standard GB 50137-2011.
        - limit (int, optional): 最多返回的aoi数量，按距离排序，近的优先（默认None）. The maximum number of AOIs returned, sorted by distance, closest ones first (default to None).

        Returns:
        - List[Tuple[Any, float]]: aoi列表，每个元素为（aoi, 距离）。aoi list, each element is (aoi, distance).
        """

        if not isinstance(center, Point):
            center = Point(center)
        # 获取半径内的aoi
        indices = self._aoi_tree.query(center.buffer(radius))
        # 过滤掉不满足城市用地条件的aoi
        aois = []
        for index in indices:
            aoi = self._aoi_list[index]
            if (
                urban_land_uses is not None
                and aoi["urban_land_use"] not in urban_land_uses
            ):
                continue
            distance = center.distance(aoi["shapely_xy"])
            aois.append((aoi, distance))
        # 按照距离排序
        aois = sorted(aois, key=lambda x: x[1])
        if limit is not None:
            aois = aois[:limit]
        return aois
