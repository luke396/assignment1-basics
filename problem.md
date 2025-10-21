# Problem

> ord('a') -> 97
> chr(97) -> 'a'

## unicode1

a. chr(0) is return '\x00'
b. print it return nothing, the `__repr__()` is '\x00' as what we have seen above
c.

```python
>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

## unicode 2

a. Using utf-8 instead of utf-16 or utf-32, because utf-8 provides shorter int list.
b. Because the code using `bytes([b]).decode`, which assume that any single bytes can be decodes,
but, '你好' 's encode can't directly decode back for just a single bytes.

```python
>>> '你'.encode('utf-8')
b'\xe4\xbd\xa0'
>>> wrong('你好'.encode('utf-8'))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in wrong
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```

c.

```python
>>> bytes([228]).decode('utf-8')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
>>> bytes([228, 189]).decode('utf-8')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 0-1: unexpected end of data
>>> bytes([228, 189, 160]).decode('utf-8')
```

## train_bpe

[train_bpe](cs336_basics/bpe.py)

## train_bpe_tinystories

(a)

Training our bpe model on TinyStories dataset with vocab size 10000, will cost 1m45s. Reading the result, the longest tokens below, and they are totally make sense.

```shell
Top 5 longest tokens (by bytes):
  1) id=7168, len=15 bytes, value=b' accomplishment' (hex=206163636f6d706c6973686d656e74)
  2) id=9152, len=15 bytes, value=b' disappointment' (hex=206469736170706f696e746d656e74)
  3) id=9388, len=15 bytes, value=b' responsibility' (hex=20726573706f6e736962696c697479)
  4) id=3236, len=14 bytes, value=b' uncomfortable' (hex=20756e636f6d666f727461626c65)
  5) id=3524, len=14 bytes, value=b' compassionate' (hex=20636f6d70617373696f6e617465)
```

(b)

```shell
uv run py-spy record -o bpe_profile.svg -- python cs336_basics/bpe.py
```

![bpe_profile.svg](./output/bpe_profile.svg)

> We get the fire graph , and we can see that the most time consuming part is `_apply_merge` function, which is updating token count using increase method.
> Before this, the most time comsuming part is file transfer in multiprocessing, which we can optimize by transferring the `start` and `end` index instead of the whole text chunk.

After some optimization of `apply_merge` using sub_tokens instead of whole tokens, optimize `_select_most_frequent_pair` using heapq, and optimize the multiprocessing file transfer, we finally get the fire graph, all parts time consuming are relatively small and balanced.

> fire graph's time consuming is including all the subprocesses' time consuming. Not the system time only.
> Actually the most time consuming part is `_process_range_for_pretokenization` function, which is reading file and pre-tokenizing. Using `scalene` we can check the system time.

## train_bpe_expts_owt

This is a really big dataset. My local machine with 16GB RAM can't handle it. Some optimization likes stream loading pre-tokens, but the implementation is really complex. So I just run it on a cloud server with more RAM.

## tokenizer
