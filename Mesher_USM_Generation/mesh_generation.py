dem_filename="path/to/DEM.tif"
errormetric    = "rmse"
max_tolerance  = 2.0      # meters; smaller = finer mesh; -1 => uniform
min_area       = 10.0      # m^2
max_area= 99999999**2  #Effectively unlimited upper area -- allow tolerance check to refine it further


constraints = { 'river_network' :
                                        {
                                           'file': 'Path/to/River_Centerline.shp',
                                           'simplify':2.0 # will be in original projection units
                                        }
                        }

simplify=True
simplify_tol=10

lloyd_itr=2
MPI_nworkers = 1
user_output_dir = "./mesh_out"
