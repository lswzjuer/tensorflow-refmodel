from kaffe.tensorflow import Network

class LIDAR(Network):
    def setup(self):
        (self.feed('data')
             .conv(3, 3, 32, 2, 2, name='resnetv1pyr1_hybridsequential0_conv0_fwd')
             .conv(3, 3, 32, 1, 1, name='resnetv1pyr1_hybridsequential1_conv0_fwd')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_hybridsequential2_conv0_fwd')
             .max_pool(3, 3, 2, 2, name='pool0_fwd')
             .conv(1, 1, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv10_hybridsequential0_conv0_fwd')
             .conv(3, 3, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv10_hybridsequential1_conv0_fwd')
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_stage_4bottleneckv10_hybridsequential2_conv0_fwd'))

        (self.feed('pool0_fwd')
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_stage_4_shortcuthybridsequential0_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_4bottleneckv10_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_4_shortcuthybridsequential0_conv0_fwd')
             .add(name='resnetv1pyr1_stage_4bottleneckv10__plus0')
             .relu(name='resnetv1pyr1_stage_4bottleneckv10_relu2_fwd')
             .conv(1, 1, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv11_hybridsequential0_conv0_fwd')
             .conv(3, 3, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv11_hybridsequential1_conv0_fwd')
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_stage_4bottleneckv11_hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_4bottleneckv11_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_4bottleneckv10_relu2_fwd')
             .add(name='resnetv1pyr1_stage_4bottleneckv11__plus0')
             .relu(name='resnetv1pyr1_stage_4bottleneckv11_relu2_fwd')
             .conv(1, 1, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv12_hybridsequential0_conv0_fwd')
             .conv(3, 3, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv12_hybridsequential1_conv0_fwd')
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_stage_4bottleneckv12_hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_4bottleneckv12_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_4bottleneckv11_relu2_fwd')
             .add(name='resnetv1pyr1_stage_4bottleneckv12__plus0')
             .relu(name='resnetv1pyr1_stage_4bottleneckv12_relu2_fwd')
             .conv(1, 1, 128, 2, 2, name='resnetv1pyr1_stage_8bottleneckv10_hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_8bottleneckv10_hybridsequential1_conv0_fwd')
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8bottleneckv10_hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_4bottleneckv12_relu2_fwd')
             .avg_pool(2, 2, 2, 2, name='resnetv1pyr1_stage_8_shortcutpool0_fwd')
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8_shortcuthybridsequential0_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_8bottleneckv10_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_8_shortcuthybridsequential0_conv0_fwd')
             .add(name='resnetv1pyr1_stage_8bottleneckv10__plus0')
             .relu(name='resnetv1pyr1_stage_8bottleneckv10_relu2_fwd')
             .conv(1, 1, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv11_hybridsequential0_conv0_fwd')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv11_hybridsequential1_conv0_fwd')
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8bottleneckv11_hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_8bottleneckv11_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_8bottleneckv10_relu2_fwd')
             .add(name='resnetv1pyr1_stage_8bottleneckv11__plus0')
             .relu(name='resnetv1pyr1_stage_8bottleneckv11_relu2_fwd')
             .conv(1, 1, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv12_hybridsequential0_conv0_fwd')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv12_hybridsequential1_conv0_fwd')
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8bottleneckv12_hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_8bottleneckv12_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_8bottleneckv11_relu2_fwd')
             .add(name='resnetv1pyr1_stage_8bottleneckv12__plus0')
             .relu(name='resnetv1pyr1_stage_8bottleneckv12_relu2_fwd')
             .conv(1, 1, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv13_hybridsequential0_conv0_fwd')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv13_hybridsequential1_conv0_fwd')
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8bottleneckv13_hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_8bottleneckv13_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_8bottleneckv12_relu2_fwd')
             .add(name='resnetv1pyr1_stage_8bottleneckv13__plus0')
             .relu(name='resnetv1pyr1_stage_8bottleneckv13_relu2_fwd')
             .conv(1, 1, 128, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential6_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_8bottleneckv13_relu2_fwd')
             .conv(1, 1, 256, 2, 2, name='resnetv1pyr1_stage_16bottleneckv10_hybridsequential0_conv0_fwd')
             .conv(3, 3, 256, 1, 1, name='resnetv1pyr1_stage_16bottleneckv10_hybridsequential1_conv0_fwd')
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16bottleneckv10_hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_8bottleneckv13_relu2_fwd')
             .avg_pool(2, 2, 2, 2, name='resnetv1pyr1_stage_16_shortcutpool0_fwd')
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_shortcuthybridsequential0_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_16bottleneckv10_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16_shortcuthybridsequential0_conv0_fwd')
             .add(name='resnetv1pyr1_stage_16bottleneckv10__plus0')
             .relu(name='resnetv1pyr1_stage_16bottleneckv10_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16bottleneckv11_hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_16bottleneckv11_hybridsequential1_conv0_fwd')
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16bottleneckv11_hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_16bottleneckv11_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16bottleneckv10_relu2_fwd')
             .add(name='resnetv1pyr1_stage_16bottleneckv11__plus0')
             .relu(name='resnetv1pyr1_stage_16bottleneckv11_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16bottleneckv12_hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_16bottleneckv12_hybridsequential1_conv0_fwd')
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16bottleneckv12_hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_16bottleneckv12_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16bottleneckv11_relu2_fwd')
             .add(name='resnetv1pyr1_stage_16bottleneckv12__plus0')
             .relu(name='resnetv1pyr1_stage_16bottleneckv12_relu2_fwd')
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential4_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_16bottleneckv12_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, padding=None, name='resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential1_conv0_fwd')
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_16bottleneckv12_relu2_fwd')
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_d_shortcuthybridsequential0_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16_d_shortcuthybridsequential0_conv0_fwd')
             .add(name='resnetv1pyr1_stage_16_dbottleneckv10__plus0')
             .relu(name='resnetv1pyr1_stage_16_dbottleneckv10_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, padding=None, name='resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential1_conv0_fwd')
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16_dbottleneckv10_relu2_fwd')
             .add(name='resnetv1pyr1_stage_16_dbottleneckv11__plus0')
             .relu(name='resnetv1pyr1_stage_16_dbottleneckv11_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, padding=None, name='resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential1_conv0_fwd')
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16_dbottleneckv11_relu2_fwd')
             .add(name='resnetv1pyr1_stage_16_dbottleneckv12__plus0')
             .relu(name='resnetv1pyr1_stage_16_dbottleneckv12_relu2_fwd')
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential3_conv0_fwd'))

        (self.feed('resnetv1pyr1_hybridsequential4_conv0_fwd', 
                   'resnetv1pyr1_hybridsequential3_conv0_fwd')
             .add(name='resnetv1pyr1__plus0')
             .conv(3, 3, 128, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential5_conv0_fwd')
             .deconv(2, 2, 128, 2, 2, relu=False, name='resnetv1pyr1_upsampling0'))

        (self.feed('resnetv1pyr1_hybridsequential6_conv0_fwd', 
                   'resnetv1pyr1_upsampling0')
             .add(name='resnetv1pyr1__plus1')
             .conv(3, 3, 64, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential7_conv0_fwd')
             .deconv(2, 2, 64, 2, 2, relu=False, name='resnetv1pyr1_upsampling1'))

        (self.feed('resnetv1pyr1_stage_4bottleneckv12_relu2_fwd')
             .conv(1, 1, 64, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential8_conv0_fwd'))

        (self.feed('resnetv1pyr1_upsampling1', 
                   'resnetv1pyr1_hybridsequential8_conv0_fwd')
             .add(name='resnetv1pyr1__plus2')
             .conv(3, 3, 64, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential9_conv0_fwd')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_share_head_4hybridsequential0_conv0_fwd')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_cls_head_4hybridsequential0_conv0_fwd')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_cls_head_4hybridsequential1_conv0_fwd')
             .conv(1, 1, 4, 1, 1, relu=False, name='resnetv1pyr1_cls_head_4hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_hybridsequential7_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_share_head_8hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_cls_head_8hybridsequential0_conv0_fwd')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_cls_head_8hybridsequential1_conv0_fwd')
             .conv(1, 1, 4, 1, 1, relu=False, name='resnetv1pyr1_cls_head_8hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_hybridsequential5_conv0_fwd')
             .conv(3, 3, 256, 1, 1, name='resnetv1pyr1_share_head_16hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_cls_head_16hybridsequential0_conv0_fwd')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_cls_head_16hybridsequential1_conv0_fwd')
             .conv(1, 1, 4, 1, 1, relu=False, name='resnetv1pyr1_cls_head_16hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_share_head_4hybridsequential0_conv0_fwd')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_reg_head_4hybridsequential0_conv0_fwd')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_reg_head_4hybridsequential1_conv0_fwd')
             .conv(1, 1, 28, 1, 1, relu=False, name='resnetv1pyr1_reg_head_4hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_share_head_8hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_reg_head_8hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_reg_head_8hybridsequential1_conv0_fwd')
             .conv(1, 1, 28, 1, 1, relu=False, name='resnetv1pyr1_reg_head_8hybridsequential2_conv0_fwd'))

        (self.feed('resnetv1pyr1_share_head_16hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_reg_head_16hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_reg_head_16hybridsequential1_conv0_fwd')
             .conv(1, 1, 28, 1, 1, relu=False, name='resnetv1pyr1_reg_head_16hybridsequential2_conv0_fwd'))