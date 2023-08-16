import itertools

box_scale = (512, 512)
b_width, b_height = 164, 164
box_dict = {
            1:(5,5,b_width,b_height), 2:(174,5,b_width,b_height), 3:(343,5,b_width,b_height), 
            4:(5,174,b_width,b_height), 5:(174,174,b_width,b_height), 6:(343,174,b_width,b_height), 
            7:(5,343,b_width,b_height), 8:(174,343,b_width,b_height), 9:(343,343,b_width,b_height)
            }
two_vertical = [(1,4), (4,7), (2,5), (5,8), (3,6), (6,9)]
two_horizen = [(1,2), (2,3), (4,5), (5,6), (7,8), (8,9)]
thr_vertical = [(1,2,3), (4,5,6), (7,8,9)]
thr_horizen = [(1,4,7), (2,5,8), (3,6,9)]
twobytwo = [(1,2,4,5), (2,3,5,6), (4,5,7,8), (5,6,8,9)]
twobythree = [(1,2,4,5,7,8), (2,3,5,6,8,9)]
layouts = []

def generate_layouts():
    # Define the grid with 9 boxes as (x, y) coordinates
    # grid = [(x, y) for x in box_grid for y in box_grid]
    grid = range(1,10)
    
    layouts = []
    for i in range(2, 9):
        for combination in itertools.combinations(grid, i):
            layouts.append(list(combination))
    
    return layouts

def merge_layouts(all_layouts):
    # Define the grid with 9 boxes as (x, y) coordinates
    layouts = []
    for layout in all_layouts:
        for i in two_vertical+two_horizen+twobythree+twobytwo+thr_horizen+thr_vertical:
            if len(set(i) & set(layout)) == len(set(layout)):
                tmp = [layout] + list(set(i)-set(layout))
                layouts.append(tmp)
            
    return layouts

def layout_to_bbox(all_layouts):
    all_bboxes = []
    for layout in all_layouts:
        bboxes = []
        for la in layout:
            if isinstance(la, list):
                min_x = 1000
                min_y = 1000
                max_x = 0
                max_y = 0
                for i in la:
                    min_x = min(min_x, box_dict[i][0])
                    min_y = min(min_y, box_dict[i][1])
                    max_x = max(max_x, box_dict[i][0])
                    max_y = max(max_y, box_dict[i][1])
                bboxes.append((min_x, min_y, max_x-min_x, max_y-min_y))
                
            else:
                bboxes.append(box_dict[la])
        all_bboxes.append(bboxes)
    
    return all_bboxes
            

def sample_bboxes_gen():
    base_layouts = generate_layouts()
    all_layouts = merge_layouts(base_layouts) + base_layouts
    all_bboxes = layout_to_bbox(all_layouts)

    print(*all_bboxes, sep='\n')
    
    return all_bboxes
