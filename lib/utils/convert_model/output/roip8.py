from kaffe.tensorflow import Network

class PVAlite-roip8(Network):
    def setup(self):
        (self.feed('data')
             .conv(4, 4, 32, 2, 2, name='conv1')
             .conv(3, 3, 48, 2, 2, name='conv2')
             .conv(3, 3, 96, 2, 2, name='conv3')
             .max_pool(3, 3, 2, 2, name='inc3a_pool1')
             .conv(1, 1, 96, 1, 1, name='inc3a_conv1'))

        (self.feed('conv3')
             .conv(1, 1, 16, 1, 1, name='inc3a_conv3_1')
             .conv(3, 3, 64, 2, 2, name='inc3a_conv3_2'))

        (self.feed('conv3')
             .conv(1, 1, 16, 1, 1, name='inc3a_conv5_1')
             .conv(3, 3, 32, 1, 1, name='inc3a_conv5_2')
             .conv(3, 3, 32, 2, 2, name='inc3a_conv5_3'))

        (self.feed('inc3a_conv1', 
                   'inc3a_conv3_2', 
                   'inc3a_conv5_3')
             .concat(3, name='inc3a')
             .conv(1, 1, 96, 1, 1, name='inc3b_conv1'))

        (self.feed('inc3a')
             .conv(1, 1, 16, 1, 1, name='inc3b_conv3_1')
             .conv(3, 3, 64, 1, 1, name='inc3b_conv3_2'))

        (self.feed('inc3a')
             .conv(1, 1, 16, 1, 1, name='inc3b_conv5_1')
             .conv(3, 3, 32, 1, 1, name='inc3b_conv5_2')
             .conv(3, 3, 32, 1, 1, name='inc3b_conv5_3'))

        (self.feed('inc3b_conv1', 
                   'inc3b_conv3_2', 
                   'inc3b_conv5_3')
             .concat(3, name='inc3b')
             .conv(1, 1, 96, 1, 1, name='inc3c_conv1'))

        (self.feed('inc3b')
             .conv(1, 1, 16, 1, 1, name='inc3c_conv3_1')
             .conv(3, 3, 64, 1, 1, name='inc3c_conv3_2'))

        (self.feed('inc3b')
             .conv(1, 1, 16, 1, 1, name='inc3c_conv5_1')
             .conv(3, 3, 32, 1, 1, name='inc3c_conv5_2')
             .conv(3, 3, 32, 1, 1, name='inc3c_conv5_3'))

        (self.feed('inc3c_conv1', 
                   'inc3c_conv3_2', 
                   'inc3c_conv5_3')
             .concat(3, name='inc3c')
             .conv(1, 1, 96, 1, 1, name='inc3d_conv1'))

        (self.feed('inc3c')
             .conv(1, 1, 16, 1, 1, name='inc3d_conv3_1')
             .conv(3, 3, 64, 1, 1, name='inc3d_conv3_2'))

        (self.feed('inc3c')
             .conv(1, 1, 16, 1, 1, name='inc3d_conv5_1')
             .conv(3, 3, 32, 1, 1, name='inc3d_conv5_2')
             .conv(3, 3, 32, 1, 1, name='inc3d_conv5_3'))

        (self.feed('inc3d_conv1', 
                   'inc3d_conv3_2', 
                   'inc3d_conv5_3')
             .concat(3, name='inc3d')
             .conv(1, 1, 96, 1, 1, name='inc3e_conv1'))

        (self.feed('inc3d')
             .conv(1, 1, 16, 1, 1, name='inc3e_conv3_1')
             .conv(3, 3, 64, 1, 1, name='inc3e_conv3_2'))

        (self.feed('inc3d')
             .conv(1, 1, 16, 1, 1, name='inc3e_conv5_1')
             .conv(3, 3, 32, 1, 1, name='inc3e_conv5_2')
             .conv(3, 3, 32, 1, 1, name='inc3e_conv5_3'))

        (self.feed('inc3e_conv1', 
                   'inc3e_conv3_2', 
                   'inc3e_conv5_3')
             .concat(3, name='inc3e')
             .max_pool(3, 3, 1, 1, name='inc4a_pool1')
             .conv(1, 1, 128, 1, 1, name='inc4a_conv1'))

        (self.feed('inc3e')
             .conv(1, 1, 32, 1, 1, name='inc4a_conv3_1')
             .conv(3, 3, 96, 1, 1, padding=None, name='inc4a_conv3_2'))

        (self.feed('inc3e')
             .conv(1, 1, 16, 1, 1, name='inc4a_conv5_1')
             .conv(3, 3, 32, 1, 1, padding=None, name='inc4a_conv5_2')
             .conv(3, 3, 32, 1, 1, padding=None, name='inc4a_conv5_3'))

        (self.feed('inc4a_conv1', 
                   'inc4a_conv3_2', 
                   'inc4a_conv5_3')
             .concat(3, name='inc4a')
             .conv(1, 1, 128, 1, 1, name='inc4b_conv1'))

        (self.feed('inc4a')
             .conv(1, 1, 32, 1, 1, name='inc4b_conv3_1')
             .conv(3, 3, 96, 1, 1, padding=None, name='inc4b_conv3_2'))

        (self.feed('inc4a')
             .conv(1, 1, 16, 1, 1, name='inc4b_conv5_1')
             .conv(3, 3, 32, 1, 1, padding=None, name='inc4b_conv5_2')
             .conv(3, 3, 32, 1, 1, padding=None, name='inc4b_conv5_3'))

        (self.feed('inc4b_conv1', 
                   'inc4b_conv3_2', 
                   'inc4b_conv5_3')
             .concat(3, name='inc4b')
             .conv(1, 1, 128, 1, 1, name='inc4c_conv1'))

        (self.feed('inc4b')
             .conv(1, 1, 32, 1, 1, name='inc4c_conv3_1')
             .conv(3, 3, 96, 1, 1, padding=None, name='inc4c_conv3_2'))

        (self.feed('inc4b')
             .conv(1, 1, 16, 1, 1, name='inc4c_conv5_1')
             .conv(3, 3, 32, 1, 1, padding=None, name='inc4c_conv5_2')
             .conv(3, 3, 32, 1, 1, padding=None, name='inc4c_conv5_3'))

        (self.feed('inc4c_conv1', 
                   'inc4c_conv3_2', 
                   'inc4c_conv5_3')
             .concat(3, name='inc4c')
             .conv(1, 1, 128, 1, 1, name='inc4d_conv1'))

        (self.feed('inc4c')
             .conv(1, 1, 32, 1, 1, name='inc4d_conv3_1')
             .conv(3, 3, 96, 1, 1, padding=None, name='inc4d_conv3_2'))

        (self.feed('inc4c')
             .conv(1, 1, 16, 1, 1, name='inc4d_conv5_1')
             .conv(3, 3, 32, 1, 1, padding=None, name='inc4d_conv5_2')
             .conv(3, 3, 32, 1, 1, padding=None, name='inc4d_conv5_3'))

        (self.feed('inc4d_conv1', 
                   'inc4d_conv3_2', 
                   'inc4d_conv5_3')
             .concat(3, name='inc4d')
             .conv(1, 1, 128, 1, 1, name='inc4e_conv1'))

        (self.feed('inc4d')
             .conv(1, 1, 32, 1, 1, name='inc4e_conv3_1')
             .conv(3, 3, 96, 1, 1, padding=None, name='inc4e_conv3_2'))

        (self.feed('inc4d')
             .conv(1, 1, 16, 1, 1, name='inc4e_conv5_1')
             .conv(3, 3, 32, 1, 1, padding=None, name='inc4e_conv5_2')
             .conv(3, 3, 32, 1, 1, padding=None, name='inc4e_conv5_3'))

        (self.feed('inc4e_conv1', 
                   'inc4e_conv3_2', 
                   'inc4e_conv5_3')
             .concat(3, name='inc4e'))

        (self.feed('conv3')
             .max_pool(3, 3, 2, 2, name='downsample'))

        (self.feed('downsample', 
                   'inc3e', 
                   'inc4e')
             .concat(3, name='concat')
             .conv(1, 1, 256, 1, 1, name='convf')
             .conv(1, 1, 256, 1, 1, name='rpn_conv1')
             .conv(1, 1, 98, 1, 1, relu=False, name='rpn_cls_score_fabu'))

        (self.feed('rpn_conv1')
             .conv(1, 1, 196, 1, 1, relu=False, name='rpn_bbox_pred_fabu'))