import copy
import unittest
import numpy as np

from camera_geometry import pixel_to_plane,world_to_pixel
from person_localization import refine_person_views,triangulate_pair


def calibration(name,origin,target):
    origin=np.asarray(origin,dtype=float);back=origin-np.asarray(target,dtype=float);back/=np.linalg.norm(back)
    right=np.cross([0.,0.,1.],back);right/=np.linalg.norm(right)
    up=np.cross(back,right);transform=np.eye(4);transform[:3,:3]=np.column_stack((right,up,back));transform[:3,3]=origin
    return dict(id=name,intrinsics=[[1200.,0.,1920.],[0.,1200.,1080.],[0.,0.,1.]],transform=transform.tolist())


def detection(c,xy,height,visible_bottom_z=0.):
    k=np.asarray(c['intrinsics']);t=np.asarray(c['transform'])
    top=world_to_pixel([*xy,height],k,t);bottom=world_to_pixel([*xy,visible_bottom_z],k,t)
    box=[top[0]-12,top[1],top[0]+12,bottom[1]]
    floor=pixel_to_plane([top[0],bottom[1]],k,t,0.)
    return dict(bbox_xyxy=box,confidence=.9,position_m=floor.tolist())


class PersonLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.a=calibration('A',[-19,0,10],[0,0,0])
        self.b=calibration('B',[19,0,10],[0,0,0])

    def views(self,height=1.75):
        return [(None,self.a,dict(persons=[detection(self.a,[.8,.8],height,.5)],unlocalized_person_count=0)),
                (None,self.b,dict(persons=[detection(self.b,[.8,.8],height)],unlocalized_person_count=0))]

    def test_occluded_feet_recover_from_two_top_rays_without_fixed_height(self):
        for height in (1.,1.6,1.9,2.2):
            with self.subTest(height=height):
                views=self.views(height);original=copy.deepcopy(views)
                refined=refine_person_views(views)
                for _,_,result in refined:
                    person=result['persons'][0]
                    np.testing.assert_allclose(person['position_m'],[.8,.8,0.],atol=1e-10)
                    self.assertEqual(person['localization_method'],'dual_view_head_triangulation')
                self.assertEqual(views,original)

    def test_single_view_keeps_detection_and_unknown_count(self):
        views=self.views()[:1];views[0][2]['unlocalized_person_count']=1
        result=refine_person_views(views)[0][2]
        self.assertEqual(result['persons'][0]['position_m'],views[0][2]['persons'][0]['position_m'])
        self.assertEqual(result['unlocalized_person_count'],1)
        self.assertEqual(result['persons'][0]['localization_method'],'floor_projection')

    def test_ambiguous_correspondence_preserves_all_original_detections(self):
        views=self.views();views[0][2]['persons']*=2
        refined=refine_person_views(views)
        self.assertEqual([len(v[2]['persons']) for v in refined],[2,1])
        self.assertTrue(all(p['localization_method']=='floor_projection' for _,_,v in refined for p in v['persons']))

    def test_two_distinct_people_match_without_crossing_identities(self):
        persons=[([.8,-3.],1.6),([-.5,3.],1.9)]
        views=[(None,c,dict(persons=[detection(c,xy,h) for xy,h in persons],unlocalized_person_count=0)) for c in (self.a,self.b)]
        views[1][2]['persons'].reverse()
        refined=refine_person_views(views)
        for view in refined:
            points=sorted(p['position_m'] for p in view[2]['persons'])
            np.testing.assert_allclose(points,[[-.5,3.,0.],[.8,-3.,0.]],atol=1e-10)

    def test_parallel_camera_rays_cannot_create_a_pair(self):
        p=detection(self.a,[.8,.8],1.75)
        self.assertIsNone(triangulate_pair(p,p,self.a,self.a))

    def test_missing_and_invalid_boxes_do_not_create_locations(self):
        valid=detection(self.b,[.8,.8],1.75)
        for box in ([],[0,1,0,3],[0,float('nan'),3,5]):
            with self.subTest(box=box):
                self.assertIsNone(triangulate_pair(dict(position_m=[0,0,0],bbox_xyxy=box),valid,self.a,self.b))


if __name__=='__main__':unittest.main()
