from .config import Config, MapConfig

from . import sao_paulo, beijing, dahongmen, paris, newyork, cape_town, london, moscow, mumbai, nairobi, san_francisco, shanghai, sydney, tokyo, wangjing, yuyuantan

all_config = {
    "beijing": beijing.config,
    "dahongmen": dahongmen.config,
    "paris": paris.config,
    "newyork": newyork.config,
    "sao_paulo":sao_paulo.config,
    "cape_town":cape_town.config, 
    "london":london.config, 
    "moscow":moscow.config, 
    "mumbai":mumbai.config, 
    "nairobi":nairobi.config, 
    "san_francisco":san_francisco.config, 
    "shanghai":shanghai.config, 
    "sydney":sydney.config, 
    "tokyo":tokyo.config,
    "wangjing":wangjing.config,
    "yuyuantan":yuyuantan.config
}
