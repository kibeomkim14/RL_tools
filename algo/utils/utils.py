import argparse

class NetkwargAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        allowed_kws = ['dim_in', 'dim_hidden', 'dim_out', 'num_layer', 'kernel', 'stride', 'dropout']
        keyword_dict = {}

        for arg in values:  #values => The args found for keyword_args
            pieces = arg.split('=')
            if len(pieces) == 2 and pieces[0] in allowed_kws:
                keyword_dict[pieces[0]] = int(pieces[1])
            else: #raise an error                                                         
                msg_inserts = ['{}='] * len(allowed_kws)
                msg_template = 'Example usage: Only {} allowed.'.format(', '.join(msg_inserts))
                msg = msg_template.format(*allowed_kws)
                raise argparse.ArgumentTypeError(msg)

        setattr(namespace, self.dest, keyword_dict) #The dest key specified in the
                                                    #parser gets assigned the keyword_dict--in
                                                    #this case it defaults to 'keyword_args'
