import math


num_bboxes = len(input_data['boxes']);


feature_downscale = 32
anchor_box_scales = [128,256,512]
anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]

output_width = 25
output_height = 25

n_anchratios = len(anchor_box_ratios)

#anchor = anchor_box_scales[0] * anchor_box_ratios[0][0]

for anchor_s_id in range(len(anchor_box_scales)):
	for anchor_r_id in range(len(anchor_box_ratios)):
		anchor_x = anchor_box_scales[anchor_s_id] * anchor_box_ratios[anchor_r_id][0]
		anchor_y = anchor_box_scales[anchor_s_id] * anchor_box_ratios[anchor_r_id][1]

		for ix in range(output_width):
			x1_anc = feature_downscale * (ix + 0.5) - anchor_x / 2
			x2_anc = feature_downscale * (ix + 0.5) + anchor_x / 2

		if x1_anc < 0 or x2_anc > resized_width:
			continue

		for jy in range(output_height):
			# y-coordinates of the current anchor box
			y1_anc = downscale * (jy + 0.5) - anchor_y / 2
			y2_anc = downscale * (jy + 0.5) + anchor_y / 2

			# ignore boxes that go across image boundaries
			if y1_anc < 0 or y2_anc > resized_height:
				continue

			# bbox_type indicates whether an anchor should be a target 
			bbox_type = 'neg'

			# this is the best IOU for the (x,y) coord and the current anchor
			# note that this is different from the best IOU for a GT bbox
			best_iou_for_loc = 0.0

			for bbox_num in range(num_bboxes):
				curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])

				if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
					cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
					cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
					cxa = (x1_anc + x2_anc)/2.0
					cya = (y1_anc + y2_anc)/2.0

					tx = (cx - cxa) / (x2_anc - x1_anc)
					ty = (cy - cya) / (y2_anc - y1_anc)
					tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
					th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
				
				if img_data['bboxes'][bbox_num]['class'] != 'bg':

					# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
					if curr_iou > best_iou_for_bbox[bbox_num]:
						best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_r_id, anchor_s_id]
						best_iou_for_bbox[bbox_num] = curr_iou
						best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
						best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

					# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
					if curr_iou > C.rpn_max_overlap:
						bbox_type = 'pos'
						num_anchors_for_bbox[bbox_num] += 1
						# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
						if curr_iou > best_iou_for_loc:
							best_iou_for_loc = curr_iou
							best_regr = (tx, ty, tw, th)

					# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
					if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
						# gray zone between neg and pos
						if bbox_type != 'pos':
							bbox_type = 'neutral'

			# turn on or off outputs depending on IOUs
			if bbox_type == 'neg':
				y_is_box_valid[jy, ix, anchor_r_id + n_anchratios * anchor_s_id] = 1
				y_rpn_overlap[jy, ix, anchor_r_id + n_anchratios * anchor_s_id] = 0
			elif bbox_type == 'neutral':
				y_is_box_valid[jy, ix, anchor_r_id + n_anchratios * anchor_s_id] = 0
				y_rpn_overlap[jy, ix, anchor_r_id + n_anchratios * anchor_s_id] = 0
			elif bbox_type == 'pos':
				y_is_box_valid[jy, ix, anchor_r_id + n_anchratios * anchor_s_id] = 1
				y_rpn_overlap[jy, ix, anchor_r_id + n_anchratios * anchor_s_id] = 1
				start = 4 * (anchor_r_id + n_anchratios * anchor_s_id)
				y_rpn_regr[jy, ix, start:start+4] = best_regr

for idx in range(num_anchors_for_bbox.shape[0]):
		if num_anchors_for_bbox[idx] == 0:
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1:
				continue
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)		#change dim to [1,9,h,w]

	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0) 	#change dim to [1,9,h,w]

	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

	num_pos = len(pos_locs[0])


	num_regions = 256

	if len(pos_locs[0]) > num_regions/2:
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
	#???
	#y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr)