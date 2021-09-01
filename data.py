
import numpy as np
import ast

class dataset():


    def generate_train_validation(self,validation,data):


        self.data=data
        validation = int(validation * len(data))
        # print(validation)
        # test = int(test * len(self))
        train = len(data) - validation 

        self.all_idx = np.random.permutation(len(self.data))
        self.train_idx = data[0:train]
        print(len(self.train_idx))
        self.validation_idx = data[train:]
        # self.test_idx = self.all_idx[train + validation:]

        self.train_counter = 0
        self.validation_counter = 0
        # self.test_counter = 0

        self.train_count = train
        self.validation_count = validation
        # self.test_count = test

    def _next_batch(self, counter, count, idx, batch_size):
        dic={}
        if batch_size is not None:
            
            if counter + batch_size >= count:
                counter = 0
                np.random.shuffle(idx)
        
            output=idx[counter:counter + batch_size]

            counter+=batch_size
            dic[0]=output
            dic[1]=counter
        else:
            dic[0]=idx
        # print(counter)
        # print(output)
        # if batch_size is not None:
        #     if counter + batch_size >= count:
        #         counter = 0
        #         np.random.shuffle(idx)

        #     output = [obj[idx[counter:counter + batch_size]]
        #               for obj in (self.data)]

        #     counter += batch_size
        # else:
        #     output = [obj[idx] for obj in (self.data)]

        return dic

    def next_train_batch(self, batch_size=None):
        out = self._next_batch(counter=self.train_counter, count=self.train_count,
                               idx=self.train_idx, batch_size=batch_size)
        self.train_counter = out[1]
        # print(self.train_counter)
        

        return out[0]

    def next_valid_batch(self, batch_size=None):
        out = self._next_batch(counter=self.validation_counter, count=self.validation_count,
                               idx=self.validation_idx, batch_size=None)
        return out[0]
        # self.train_counter = out[0]












