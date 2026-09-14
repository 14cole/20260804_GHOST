! Persistent source-only Helmholtz plan for GHOST.
! Calls the pinned, unmodified FMM2D tree and evaluation routines. Tree, source
! ordering, expansion sizes and workspaces survive changes of density. Native
! calls (including create/destroy) are serialized by the Python binding.
module ghost_plan_module
  use iso_c_binding
  implicit none
  type plan
    integer :: n, levels, boxes, ltree, ndiv, nmax, nd = 0, lmptot, lmptmp
    integer :: iptr(8)
    real(c_double) :: eps
    complex(c_double_complex) :: zk
    integer, allocatable :: tree(:), order(:), srcse(:,:), targse(:,:), expse(:,:)
    integer, allocatable :: terms(:), addr(:,:)
    real(c_double), allocatable :: sizes(:), scales(:), centers(:,:), xy(:,:)
    real(c_double), allocatable :: work(:), directions(:,:,:)
    complex(c_double_complex), allocatable :: temp(:), charge(:,:), dipole(:,:)
    complex(c_double_complex), allocatable :: pot(:,:), grad(:,:,:)
  end type
  type slot
    type(plan), pointer :: p => null()
    integer(c_int64_t) :: id = 0
  end type
  type(slot), save :: slots(256)
  integer(c_int64_t), save :: sequence = 0
contains
  function plan_bytes(p) result(nbytes)
    type(plan), intent(in) :: p
    integer(c_int64_t) :: nbytes
    nbytes = 4_c_int64_t * (p%ltree + p%n + 6*p%boxes + p%levels+1) &
         + 8_c_int64_t * (2*(p%levels+1) + 2*p%boxes + 2*p%n)
    if (p%nd > 0) nbytes = nbytes + 16_c_int64_t*p%boxes + 8_c_int64_t*p%lmptot &
         + 16_c_int64_t*p%lmptmp + 96_c_int64_t*p%nd*p%n
  end function

  subroutine ghost_fmm_create(n, xy, zk, eps, handle, nbytes, ier) bind(C)
    integer(c_int), intent(in) :: n
    real(c_double), intent(in) :: xy(2,n), eps
    complex(c_double_complex), intent(in) :: zk
    integer(c_int64_t), intent(out) :: handle, nbytes
    integer(c_int), intent(out) :: ier
    type(plan), pointer :: p
    real(c_double) :: dummy(2,1), pi
    integer :: s, idiv, i, status
    handle=0; nbytes=0; ier=1
    if (n < 1) return
    do s=1,size(slots)
      if (.not.associated(slots(s)%p)) exit
    enddo
    if (s > size(slots)) return
    allocate(slots(s)%p, stat=status)
    if (status /= 0) return
    p => slots(s)%p
    p%n=n; p%zk=zk; p%eps=eps
    call hndiv2d(eps,n,0,1,1,2,0,p%ndiv,idiv)
    call pts_tree_mem(xy,n,dummy,0,idiv,p%ndiv,0,51,0,0,p%levels,p%boxes,p%ltree)
    allocate(p%tree(p%ltree),p%sizes(0:p%levels),p%centers(2,p%boxes), &
         p%order(n),p%srcse(2,p%boxes),p%targse(2,p%boxes),p%expse(2,p%boxes), &
         p%xy(2,n),p%scales(0:p%levels),p%terms(0:p%levels),stat=status)
    if (status /= 0) then
      deallocate(slots(s)%p); nullify(slots(s)%p); return
    endif
    call pts_tree_build(xy,n,dummy,0,idiv,p%ndiv,0,51,0,0,p%levels,p%boxes, &
         p%ltree,p%tree,p%iptr,p%centers,p%sizes)
    call pts_tree_sort(n,xy,p%tree,p%ltree,p%boxes,p%levels,p%iptr,p%centers,p%order,p%srcse)
    p%targse(1,:)=1; p%targse(2,:)=0
    p%expse(1,:)=1; p%expse(2,:)=0
    call dreorderf(2,n,xy,p%xy,p%order)
    pi=4*atan(1.d0); p%nmax=0
    do i=0,p%levels
      p%scales(i)=min(abs(zk*p%sizes(i)/(2*pi)),1.d0)
      call h2dterms(p%sizes(i),zk,eps,p%terms(i),ier)
      if (ier /= 0) then
        deallocate(slots(s)%p); nullify(slots(s)%p); return
      endif
      p%nmax=max(p%nmax,p%terms(i))
    enddo
    sequence=sequence+1
    slots(s)%id=sequence*size(slots)+s
    handle=slots(s)%id; nbytes=plan_bytes(p); ier=0
  end subroutine

  subroutine ghost_fmm_apply(handle,nd,ic,charge,idp,dipole,directions,pg,pot,grad,nbytes,ier) bind(C)
    integer(c_int64_t), intent(in) :: handle
    integer(c_int), intent(in) :: nd,ic,idp,pg
    complex(c_double_complex), intent(in) :: charge(nd,*),dipole(nd,*)
    real(c_double), intent(in) :: directions(nd,2,*)
    complex(c_double_complex), intent(out) :: pot(nd,*),grad(nd,2,*)
    integer(c_int64_t), intent(out) :: nbytes
    integer(c_int), intent(out) :: ier
    type(plan), pointer :: p
    integer :: s,status
    real(c_double) :: dummy(2,1), times(8), scj
    complex(c_double_complex) :: cdummy(nd,3,1), jexps(100)
    s=int(modulo(handle-1, int(size(slots),c_int64_t)))+1
    ier=1; nbytes=0
    if (.not.associated(slots(s)%p)) return
    if (slots(s)%id /= handle) return
    if (nd < 1 .or. pg < 1 .or. pg > 2) return
    p => slots(s)%p
    if (p%nd /= nd) then
      if (p%nd > 0) deallocate(p%addr,p%work,p%temp,p%charge,p%dipole,p%directions,p%pot,p%grad)
      p%nd=0
      allocate(p%addr(4,p%boxes),stat=status)
      if (status /= 0) return
      call h2dmpalloc(nd,p%tree(p%iptr(1)),p%addr,p%levels,p%lmptot,p%terms)
      p%lmptmp=(2*p%nmax+1)*nd
      allocate(p%work(p%lmptot),p%temp(p%lmptmp),p%charge(nd,p%n), &
           p%dipole(nd,p%n),p%directions(nd,2,p%n),p%pot(nd,p%n),p%grad(nd,2,p%n),stat=status)
      if (status /= 0) then
        ! A failed plan cannot be reused. Release every partially allocated array.
        deallocate(slots(s)%p); nullify(slots(s)%p); slots(s)%id=0
        return
      endif
      p%nd=nd
    endif
    if (ic == 1) call dreorderf(2*nd,p%n,charge,p%charge,p%order)
    if (idp == 1) then
      call dreorderf(2*nd,p%n,dipole,p%dipole,p%order)
      call dreorderf(2*nd,p%n,directions,p%directions,p%order)
    endif
    p%pot=0; p%grad=0; times=0; ier=0; dummy=0; cdummy=0; scj=0; jexps=0
    call hfmm2dmain(nd,p%eps,p%zk,p%n,p%xy,ic,p%charge,idp,p%dipole,p%directions, &
         0,dummy,0,dummy,p%addr,p%work,p%temp,p%lmptmp,p%tree,p%ltree,p%iptr, &
         p%ndiv,p%levels,p%boxes,0,p%sizes,p%scales,p%centers,p%tree(p%iptr(1)), &
         p%srcse,p%targse,p%expse,p%terms,0,pg,p%pot,p%grad,cdummy, &
         0,cdummy,cdummy,cdummy,jexps,scj,1,times,ier)
    if (ier /= 0) return
    call dreorderi(2*nd,p%n,p%pot,pot,p%order)
    if (pg == 2) call dreorderi(4*nd,p%n,p%grad,grad,p%order)
    nbytes=plan_bytes(p)
  end subroutine

  subroutine ghost_fmm_destroy(handle) bind(C)
    integer(c_int64_t), intent(in) :: handle
    integer :: s
    s=int(modulo(handle-1, int(size(slots),c_int64_t)))+1
    if (slots(s)%id /= handle) return
    if (associated(slots(s)%p)) deallocate(slots(s)%p)
    nullify(slots(s)%p); slots(s)%id=0
  end subroutine
end module
